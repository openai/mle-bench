import logging
import random
from typing import Any, Callable, cast
import json
import humanize
import subprocess
import shutil
import math
import os
from typing import Optional
from aide.prompt import get_impl_guideline_from_agent
from openai import OpenAI
import re
from pathlib import Path
from .backend import FunctionSpec, query
from .interpreter import ExecutionResult
from .journal import Journal, Node
from .utils import data_preview
from .utils.config import Config
from .utils.metric import MetricValue, WorstMetricValue
from .utils.response import extract_code, extract_text_up_to_code, wrap_code
from .utils.bm25_faiss import NodeRetriever
from .utils.evaluator import evaluator
import numpy as np
from .memory.buffer import AgentMemoryManager
from .memory.global_memo import GlobalMemoryManager

logger = logging.getLogger("aide")

ExecCallbackType = Callable[[str, bool], ExecutionResult]
class Agent:
    def __init__(
            self,
            task_desc: str,
            kaggle_tool: str,
            cfg: Config,
            journal: Journal,
    ):
        super().__init__()
        self.task_desc = task_desc
        self.kaggle_tool = kaggle_tool
        self.cfg = cfg
        self.acfg = cfg.agent
        self.best_node: Node | None = None
        self.journal = journal
        self.data_preview: str | None = None
        self.evaluator = evaluator(self.cfg, self.task_desc)
        self.current_step = len(self.journal)
        self.pretrain_model_guide_path = Path("/inspire/ssd/tenant_predefaa-9a1b-4522-bb10-8850f313be13/global_user/5965-xiaokangcheng/aideml/aide/kaggle_prompt/pretrained_models_usage_guide.md")
        self.buffer_memo = AgentMemoryManager(self.cfg)
        self.global_memo = GlobalMemoryManager(self.cfg)
        with open(self.pretrain_model_guide_path , 'r', encoding='utf-8') as f:
            self.pretrain_model_guide = f.read()
        self.scot_path = Path("/inspire/ssd/tenant_predefaa-9a1b-4522-bb10-8850f313be13/global_user/5965-xiaokangcheng/aideml/aide/prompt/scot.md")
        with open(self.scot_path , 'r', encoding='utf-8') as f:
            self.scot = f.read()

        # --- Forced explode: stagnation tracking ---
        self._stagnation_best_metric: float | None = None
        self._stagnation_counter: int = 0
        self._last_forced_explode_step: int = -999  # allow first trigger immediately

    def _get_bug_priority(self, bug_node: Node) -> float:
        """
        计算 bug 节点的修复优先级分数（越高越优先修复）。

        评分维度：
        1. 父节点质量 — 父节点 metric 越好，说明 bug 只差一步就能产出好结果，优先修
        2. debug 深度 — 已修多次还没好，降低优先级（避免陷入死循环）
        3. bug 类型 — 简单错误（import/语法）优先修，复杂错误（OOM/逻辑）降低优先级
        4. 新鲜度 — 最近产生的 bug 优先修（上下文还在记忆中）
        """
        score = 0.0

        # 1. 父节点质量：如果父节点是好节点且 metric 接近 best，优先级高
        if bug_node.parent and not bug_node.parent.is_buggy:
            parent_metric = bug_node.parent.metric
            if parent_metric and parent_metric.value is not None:
                best_node = self.journal.get_best_node()
                if best_node and best_node.metric and best_node.metric.value is not None:
                    if parent_metric.maximize:
                        ratio = parent_metric.value / (best_node.metric.value + 1e-6)
                    else:
                        ratio = best_node.metric.value / (parent_metric.value + 1e-6)
                    score += 3.0 * min(ratio, 1.0)

        # 2. debug 深度惩罚：修了太多次还没好，降低优先级
        debug_depth = getattr(bug_node, "debug_depth", 0)
        score -= debug_depth * 0.8

        # 3. bug 类型评分：简单错误优先修，复杂错误降低优先级
        simple_errors = [
            "ImportError", "ModuleNotFoundError", "NameError",
            "SyntaxError", "IndentationError", "KeyError",
            "IndexError", "TypeError", "AttributeError",
            "ValueError", "FileNotFoundError"
        ]
        complex_errors = [
            "OutOfMemoryError", "CUDA", "TimeoutError",
            "RuntimeError", "MemoryError", "Killed"
        ]

        if bug_node.exc_type:
            exc_str = str(bug_node.exc_type)
            if any(e in exc_str for e in simple_errors):
                score += 2.0  # 简单错误，容易修
            elif any(e in exc_str for e in complex_errors):
                score -= 1.0  # 复杂错误，可能很难修

        # 4. 新鲜度：最近产生的 bug 优先修
        if bug_node.step is not None and len(self.journal) > 0:
            recency = 1.0 / (len(self.journal) - bug_node.step + 1)
            score += recency * 1.5

        return score

    def _check_stagnation(self) -> bool:
        """
        Check if the search has stagnated.
        Only counts steps where a GOOD (non-buggy) node was produced but didn't improve the best metric.
        Bug nodes are ignored — they are failed attempts, not stagnation.
        Returns True if forced explode should be triggered.
        """
        fe_cfg = getattr(self.acfg.search, 'forced_explode', None)
        if not fe_cfg or not getattr(fe_cfg, 'enabled', False):
            return False

        # Only check when a new node has been added
        if len(self.journal) == 0:
            return False

        latest_node = self.journal.nodes[-1]

        # Skip bug nodes — they don't count toward stagnation
        if latest_node.is_buggy or latest_node.node_type == 'deadend':
            return False

        # Cooldown check: don't trigger too frequently
        current_step = len(self.journal)
        cooldown = getattr(fe_cfg, 'cooldown', 15)
        if current_step - self._last_forced_explode_step < cooldown:
            return False

        best_node = self.journal.get_best_node()
        if not best_node or not best_node.metric or best_node.metric.value is None:
            return False

        current_best = best_node.metric.value
        maximize = best_node.metric.maximize

        if self._stagnation_best_metric is None:
            self._stagnation_best_metric = current_best
            self._stagnation_counter = 0
            return False

        # Check if improved using percentage-based threshold
        threshold_pct = getattr(fe_cfg, 'stagnation_threshold_pct', 3.0) / 100.0
        abs_threshold = abs(self._stagnation_best_metric) * threshold_pct + 1e-9
        if maximize:
            improved = current_best > self._stagnation_best_metric + abs_threshold
        else:
            improved = current_best < self._stagnation_best_metric - abs_threshold

        if improved:
            self._stagnation_best_metric = current_best
            self._stagnation_counter = 0
            return False
        else:
            # Only increment for good nodes that didn't improve
            self._stagnation_counter += 1
            patience = getattr(fe_cfg, 'patience', 8)
            if self._stagnation_counter >= patience:
                logger.info(f"[Stagnation] {self._stagnation_counter} good nodes produced without "
                           f"improving best metric ({self._stagnation_best_metric:.4f}). "
                           f"Triggering forced explode.")
                return True
            return False

    def _forced_explode(self) -> list[Node]:
        """
        Force a high-temperature, high-amplitude explosion on the best UCT node
        to break out of local optimum stagnation.
        """
        fe_cfg = self.acfg.search.forced_explode
        best_node = self.journal.get_best_node()

        if not best_node:
            logger.warning("[Forced Explode] No best node found, skipping.")
            return []

        amplitude = getattr(fe_cfg, 'amplitude', 0.95)
        num_sparks = getattr(fe_cfg, 'num_sparks', 5)
        temperature = getattr(fe_cfg, 'temperature', 1.2)

        logger.info(f"[Forced Explode] Triggering on best node {best_node.id} "
                    f"(metric={best_node.metric.value:.4f}, UCT={best_node.uct:.3f}) "
                    f"with amplitude={amplitude}, temp={temperature}, sparks={num_sparks}")

        sparks = self._explode(best_node, amplitude, num_sparks, forced_temp=temperature)

        # Reset stagnation counter and record the step
        self._stagnation_counter = 0
        self._last_forced_explode_step = len(self.journal)

        logger.info(f"[Forced Explode] Generated {len(sparks)} sparks. "
                    f"Cooldown until step {self._last_forced_explode_step + fe_cfg.cooldown}")
        return sparks

    def search_policy(self) :
        """Select a node to work on (or None to draft a new node)."""
        search_cfg = self.acfg.search
        global_step = len(self.journal)
        max_step = self.acfg.steps
        
        # 计算当前搜索进度 (0.0 -> 1.0)
        progress = min(1.0, global_step / max_step) if max_step > 0 else 0.0

        # initial drafting，若初始draft节点数比设定的上限值小
        if len(self.journal.draft_nodes) < search_cfg.num_drafts:
            logger.info("[search policy] drafting new node (not enough drafts)")
            return None

        # --- Forced explode: check for stagnation BEFORE normal policy ---
        if self._check_stagnation():
            logger.info("[search policy] STAGNATION DETECTED — forcing high-temp explode on best UCT node")
            best_node = self.journal.get_best_node()
            if best_node:
                return [best_node], "forced_explode"

        # debugging
        # 1. 识别“链式 Debug”状态：检查最后加入 Journal 的节点，如果是新的bug节点，进行一次debug
        last_node = self.journal.nodes[-1] if len(self.journal.nodes) > 0 else None
        if last_node and last_node.is_buggy and last_node.debug_depth < search_cfg.back_debug_depth:
            logger.info("[search policy] debugging")
            return [last_node], "debug"

        debuggable_nodes = [
            n for n in self.journal.buggy_nodes
            if n.is_leaf and n.debug_depth < search_cfg.max_debug_depth
        ]
        # 按优先级分数降序排序（最高优先级的 bug 排最前）
        if debuggable_nodes:
            debuggable_nodes.sort(key=lambda n: self._get_bug_priority(n), reverse=True)

        good_nodes = self.journal.good_nodes
        # --- 进入 FWA + 进化算法逻辑 ---

        weights = []
        if good_nodes:
            total_visits = sum(n.visits for n in self.journal.nodes)
            for n in good_nodes:
                # 越接近 best_score，权重越高 (用于分配火花数)
                n.uct = n.uct_value(total_forest_visits = total_visits)#计算各个好节点的uct值
            all_ucts = [n.uct for n in good_nodes]
            best_uct = max(all_ucts)
            worst_uct = min(all_ucts)
            for n in good_nodes:
                w = (n.uct - worst_uct + 1e-6) / (best_uct - worst_uct + 1e-6)
                weights.append(w)

            # --- 自适应 debug_prob：近期 bug 率高时临时提高 debug 概率 ---
            recent_window = min(20, len(self.journal))
            if recent_window > 0:
                recent_bugs = sum(1 for n in self.journal.nodes[-recent_window:] if n.is_buggy)
                bug_rate = recent_bugs / recent_window
                adaptive_debug_prob = min(0.5, search_cfg.debug_prob + bug_rate * 0.3)
            else:
                adaptive_debug_prob = search_cfg.debug_prob

            # --- 概率分配策略 ---
            # [0.0 - adaptive_debug_prob]: debug (自适应，bug多概率高)
            # [adaptive_debug_prob - 0.6]: improve (静态，固定占位)
            # [0.6 - 1.0]: explode + merge (共占0.4，随进度动态分配)
            #   - 早期: explode 权重高 (探索多样性)
            #   - 晚期: merge 权重高 (利用已有好解杂交)
            p_improve = 0.6 - adaptive_debug_prob
            explode_share = 1.0 - progress
            merge_share = progress
            total_share = explode_share + merge_share + 1e-9
            p_explode = 0.4 * (explode_share / total_share)
            p_merge = 0.4 * (merge_share / total_share)

            # 累积概率阈值
            p_debug_threshold = adaptive_debug_prob
            p_improve_threshold = p_debug_threshold + p_improve
            p_explode_threshold = p_improve_threshold + p_explode
            # p_merge 占剩余部分到 1.0

            r = random.random()

            # 1. Debug 判断
            if r < p_debug_threshold and debuggable_nodes:
                best_bug = debuggable_nodes[0]
                logger.info(f"[search policy] debugging highest priority bug: {best_bug.id}, "
                           f"priority={self._get_bug_priority(best_bug):.2f}, "
                           f"exc_type={best_bug.exc_type}, "
                           f"adaptive_debug_prob={adaptive_debug_prob:.2f}")
                return [best_bug], "debug"

            # 2. Improve 判断 (静态概率)
            if r < p_improve_threshold:
                topk_k = search_cfg.topk_best
                topk_pool = self.journal.get_topk_best_nodes(k=topk_k, list_node=good_nodes)
                best_node = self.journal.get_best_node()
                target_node = self.get_weighted_random_node(topk_pool, best_node, temperature=1.0)
                if not target_node:
                    target_node = best_node
                logger.info(f"[search policy] improving: node={target_node.id}(UCT={target_node.uct:.3f}), "
                           f"topk_pool_size={len(topk_pool)}, p_improve={p_improve:.2f}")
                return [target_node], "improve"

            # 3. Explode 判断 (早期倾向高)
            if r < p_explode_threshold:
                target_node = random.choices(good_nodes, weights=weights, k=1)[0]
                logger.info(f"[search policy] exploding: p_explode={p_explode:.2f}, p_merge={p_merge:.2f}")
                return [target_node], "explode"

            # 4. Merge 判断 (晚期倾向高)
            best_node = self.journal.get_best_node()
            topk_k = search_cfg.topk_best
            topk_pool = self.journal.get_topk_best_nodes(k=topk_k, list_node=good_nodes)
            parent2 = self.get_weighted_random_node(topk_pool, best_node, temperature=1.0)
            if not parent2:
                logger.info(f"[search policy] merge fallback: only best node in top-k, switching to improve")
                return [best_node], "improve"
            logger.info(f"[search policy] merging: parent1={best_node.id}(UCT={best_node.uct:.3f}), "
                       f"parent2={parent2.id}(UCT={parent2.uct:.3f}), "
                       f"p_explode={p_explode:.2f}, p_merge={p_merge:.2f}")
            return [best_node, parent2], "merge"
        else:
            if debuggable_nodes:
                # 此时不再看概率，按优先级修 bug（选最高优先级的）
                best_bug = debuggable_nodes[0]
                logger.info(f"[search policy] No good nodes, debugging highest priority bug: {best_bug.id}, "
                           f"priority={self._get_bug_priority(best_bug):.2f}")
                return [best_bug], "debug"
            else:
                # 如果连可 Debug 的节点都没有了（全部到了 max_debug_depth）
                # 这说明当前的几个根节点全死了，这时候必须“扩招”一个新的 Draft
                # 或者对一个 Bug 节点进行强制性的重构（不走 debug 逻辑，走 improve）
                logger.warning("[search policy] CRITICAL: All branches failed and reached depth limit.")
                # 策略：突破 draft 限制，增加一个新维度，防止死锁
                return None




    def get_weighted_random_node(self, topk_best, best_node, temperature=1.0) -> Node | None:
        # 1. 过滤掉 best_node 及其同源节点 (保持逻辑一致)
        candidates = [
            node for node in topk_best
            if node.id != best_node.id and node.root != best_node.root
        ]

        if not candidates:
            return None

        # 2. 提取 UCT 分数
        # UCT 已经包含了均值（开发）和置信区间（探索），且统一了越大越好
        # 在计算前确保 UCT 是最新的
        
        total_visits = sum(n.visits for n in self.journal.nodes)
        for node in candidates:
            node.uct = node.uct_value(total_forest_visits=total_visits)

        uct_scores = np.array([float(node.uct) for node in candidates])

        # 3. 计算概率 (Softmax)
        if len(candidates) == 1:
            return candidates[0]

        # 数值稳定性处理：如果所有候选节点的 UCT 几乎一样
        if np.std(uct_scores) < 1e-9:
            weights = [1.0] * len(candidates)
        else:
            # 使用 Softmax 转化 UCT 分数为概率权重
            # 减去最大值防止 exp 溢出
            # temperature 控制采样的“随机度”：T越小，越倾向于选UCT最高的；T越大，采样越均匀
            shifted_scores = (uct_scores - np.max(uct_scores)) / (temperature + 1e-9)
            exp_scores = np.exp(shifted_scores)
            weights = (exp_scores / np.sum(exp_scores)).tolist()

        # 4. 根据权重随机选择一个节点
        chosen_node = random.choices(candidates, weights=weights, k=1)[0]

        return chosen_node

    @property
    def _prompt_environment(self):
        pkgs = [
            "numpy",
            "pandas",
            "scikit-learn",
            "statsmodels",
            "xgboost",
            "lightGBM",
            "torch",
            "torchvision",
            "torch-geometric",
            "bayesian-optimization",
            "timm",
            "transformers",
            "nltk",
            "spacy",
        ]
        random.shuffle(pkgs)
        pkg_str = ", ".join([f"`{p}`" for p in pkgs])

        env_prompt = {
            "Installed Packages": f"Your solution can use any relevant machine learning packages such as: {pkg_str}. Feel free to use any other packages too (all packages are already installed!). For neural networks we suggest using PyTorch rather than TensorFlow."
        }
        return env_prompt

    @property
    def _prompt_impl_guideline(self):
        guideline =  get_impl_guideline_from_agent(self)
        return {"Implementation guideline": guideline}

    @property
    def _prompt_resp_fmt(self):
        return {
            "Response format": (
                "****CRITICAL：Your response should be a brief outline/sketch of your proposed solution in natural language (MUST be strictly 3-5 sentences), "
                "****CRITICAL：followed by a single markdown code block (wrapped in ```:begin with ``` and end with ```) which implements this solution and prints out the evaluation metric. "
                "There should be no additional headings or text in your response. Just natural language text (3-5 sentences total) followed by a newline and then the markdown code block. "
            )
        }

    def plan_and_code_query(self, prompt, retries=3) -> tuple[str, str]:
        """Generate a natural language plan + code in the same LLM call and split them apart."""
        last_completion_text = ""
        
        for i in range(retries):
            completion_text = query(
                system_message=prompt,
                user_message=None,
                temperature=self.acfg.draft.temp,
                model=self.acfg.draft.model,
                cfg=self.cfg, 
            )
            
            if not completion_text:
                print(f"Attempt {i+1}: LLM returned empty response.")
                continue
            
            last_completion_text = completion_text # 记录最后一次有效响应
            
            # 立即在循环内尝试提取
            code = extract_code(completion_text)
            nl_text = extract_text_up_to_code(completion_text)
    
            if code and nl_text:
                # 提取成功，直接返回，不再重试
                return nl_text, code
    
            print(f"Attempt {i+1}: Plan + code extraction failed, retrying...")
    
        # 如果所有重试都失败了
        print("Final plan + code extraction attempt failed, giving up...")
        # 建议返回空字符串或抛出异常，取决于下游逻辑
        return "", last_completion_text

    def plan_and_code_query_explode(self, prompt, temperature=None, retries=3) -> tuple[str, str]:
        """
        生成自然语言计划 + 代码。
        优先使用传入的 temperature，如果为 None 则回退到配置文件的默认值。
        """
        # 确定最终使用的 temperature
        temp = temperature if temperature is not None else self.acfg.draft.temp
        last_completion_text = ""
        for i in range(retries):
            completion_text = query(
                system_message=prompt,
                user_message=None,
                temperature=temp,
                model=self.acfg.draft.model,
                cfg=self.cfg, 
            )
            
            if not completion_text:
                print(f"Attempt {i+1}: LLM returned empty response.")
                continue
            
            last_completion_text = completion_text # 记录最后一次有效响应
            
            # 立即在循环内尝试提取
            code = extract_code(completion_text)
            nl_text = extract_text_up_to_code(completion_text)
    
            if code and nl_text:
                # 提取成功，直接返回，不再重试
                return nl_text, code
    
            print(f"Attempt {i+1}: Plan + code extraction failed, retrying...")
    
        # 如果所有重试都失败了
        print("Final plan + code extraction attempt failed, giving up...")
        # 建议返回空字符串或抛出异常，取决于下游逻辑
        return "", last_completion_text

    def format_prompt_dict(self, data: Any, level: int = 1) -> str:
        """
        将嵌套字典递归转换为带标题层级的 Markdown 文本。
        """
        if not isinstance(data, dict):
            return str(data)
        
        lines = []
        for key, value in data.items():
            # 根据递归深度生成 Markdown 标题或加粗标签
            if level <= 2:
                prefix = "#" * level
                lines.append(f"\n{prefix} {key}")
            else:
                lines.append(f"**{key}**:")
            
            # 递归处理内容
            if isinstance(value, dict):
                lines.append(self.format_prompt_dict(value, level + 1))
            elif isinstance(value, list):
                for item in value:
                    lines.append(f"- {item}")
            else:
                lines.append(str(value))
                
        return "\n".join(lines).strip()
        
    def plan_and_code_cli(self, prompt, retries=3) -> tuple[str, str]:
        """通过命令行工具 (如 qwen -p) 获取模型输出并分离 Plan 和 Code"""
        completion_text = None
        
        # 构建增强后的 Prompt (加入 CLI Manual 提升代码质量)
        if isinstance(prompt, dict):
            prompt = self.format_prompt_dict(prompt)
        print(f"DEBUG: Prompt length is {len(prompt)} characters.")
        if len(prompt) > 30000:
            print("WARNING: Prompt is extremely long, this might be causing the crash.")

        my_env = os.environ.copy()
        my_env["NODE_OPTIONS"] = "--max-old-space-size=40960"
    
        for i in range(3):
            try:
                # 确保 prompt 是字符串
                result = subprocess.run(
                    ["qwen", "-p", prompt],
                    capture_output=True,
                    text=True,
                    encoding='utf-8',
                    check=True,
                    stdin=subprocess.DEVNULL,
                )
        
                completion_text = result.stdout
                print(completion_text)
        
                if completion_text:
                    code = extract_code(completion_text)
                    nl_text = extract_text_up_to_code(completion_text)
        
                    if code:  # 如果成功提取到代码
                        print(f"--- Attempt {i + 1} Success ---")
                        return nl_text, code
                        break  # 成功后跳出循环
        
                print(f"Attempt {i + 1}: No valid code found in output.")
        
            except subprocess.CalledProcessError as e:
                print(f"CLI 调用失败 (第 {i + 1} 次): {e.stderr}")
            except FileNotFoundError:
                print("错误：找不到 'qwen' 命令，请确认已安装相关 CLI 工具。")
                break
            except Exception as e:
                print(f"发生未知错误: {e}")
    
            print(f"Plan + code extraction failed (Attempt {i+1}), retrying...")
    
        return "", completion_text or ""

    # def generate_summary_prompt(self, memory: str, tag: str, max_length: int = 300) -> str:
    #     """
    #     根据不同的工作流阶段 (tag) 生成对应的 Memory Summary Prompt。

    #     Args:
    #         memory (str): 原始的记忆内容（包括代码、计划、历史记录等字符串）。
    #         tag (str): 当前的任务标签，可选值: 'draft', 'improve', 'debug'。
    #         max_length (int): 限制生成的 Summary 的最大字数/Token数。

    #     Returns:
    #         str: 用于输入 LLM 进行总结的 Prompt。
    #     """

    #     # 基础约束：任何总结都必须简洁，且带有字数限制
    #     base_constraint = f"Please summarize the provided content within {max_length} words/tokens."
    #     if not memory:
    #         memory = "memory is null"
    #         logger.info("memory is null")
    #     logger.info(f"memory to summarize : {memory}")

    #     # 1. Draft 模式：目的是“广度探索”，避免路径重复
    #     if tag == 'draft':
    #         system_instruction = (
    #             "You are a 'Divergence Analyzer'.\n"
    #             "Input: A history of exploration nodes, where each node contains  a 'plan', a 'result' and a 'metric' or null.\n"
    #             "Goal: Summarize the distinct approaches/algorithms used in the previous plans.\n"
    #             "Critical Requirement: The summary must highlight what has ALREADY been tried so future generations can implement completely DIFFERENT architectures or logic.\n"
    #             "Output format: A concise bulleted list of 'Used Strategies'."
    #         )

    #     # 2. Improve 模式：目的是“深度优化”，避免技巧重复
    #     elif tag == 'improve':
    #         system_instruction = (
    #             "You are an 'Optimization Tracker'.\n"
    #             "Input: A code generation history, including an 'improve_mem' (history of improvement attempts).\n"
    #             "Goal: Extract the specific optimization techniques or refactoring methods already applied in the 'improve_mem'.\n"
    #             "Critical Requirement: Focus on the 'improve_mem'. Identify what techniques (e.g., caching, vectorization, changing data structures) have been used to ensure the next step does not repeat the same optimization trick.\n"
    #             "Output format: A summary of 'Applied Optimizations'."
    #         )

    #     # 3. Debug 模式：目的是“纠错”，避免重复错误
    #     elif tag == 'debug':
    #         system_instruction = (
    #             "You are a 'Root Cause Analyst'.\n"
    #             "Input: A code generation history, including bug of all nodes(bug analysis and fix plan) and a 'debug_mem' (history of debug attempts of the chain)，and the previous check to fix some basic bugs.\n"
    #             "Goal: Analyze the **Input** to identify failed fix attempts and incorrect hypotheses.\n"
    #             "Critical Requirement: Summarize the logic of the FAILED fixes. The goal is to create a 'Do Not Repeat' list of mistakes so the next debugging attempt avoids these specific pitfalls.You should **merge and summarize** the same bug reasons.\n"
    #             "Output format: A warning list of 'Failed Fixes & Invalid Hypotheses'."
    #         )

    #     else:
    #         raise ValueError(f"Unknown tag: {tag}. Expected 'draft', 'improve', or 'debug'.")

    #     # 组装最终的 Prompt
    #     final_prompt = f"""
    # {system_instruction}
    # {base_constraint}
    # """
    #     response_text = query(
    #             system_message=audit_prompt,
    #             user_message=None,
    #             temperature=self.acfg.feedback.temp,
    #             model=self.acfg.feedback.model,
    #             cfg=self.cfg
    #         )
    #     # 将结果存入 self.kaggle_tool 供 _draft 使用
    #     result = response_text.choices[0].message.content
    #     logger.info(f"Summarize the memory output is :{result}")
    #     return result
    def _draft(self) -> Node:
        prompt: dict = {
            "Introduction": (
                "You are a Kaggle Grandmaster. Your goal is to design and implement a highly competitive ML solution "
                "that learns from data and generates real predictions. Treat this with professionalism."
            ),
            
            # 核心输入信息
            "Task Description": self.task_desc,
            
            # 极简且致命的环境/代码约束（防止直接跑崩）
            "Execution Constraints": [
                "1. CRITICAL: Set `num_workers=8` in all DataLoaders.",
                "2. Hugging Face mirror: Configure via HF_ENDPOINT if downloading models.",
                "3. NO PLACEHOLDERS: The generated Python code must be 100% complete and runnable.",
                "4. Print ONLY the final validation score at the very end of the execution."
            ],
            # 新增：显式训练协议约束，强制打破“只训练一个 epoch”的幻觉
            "ATTENTION": [
                "****1. MULTI-EPOCH TRAINING: You MUST implement a full training loop with multiple epochs (e.g., 5-20 depending on task).****",
                "****2. ITERATIVE LOGIC: The `train_one_epoch` in Kaggle Blueprint is a sub-component; ensure it is called within a proper `for epoch in range(N)` loop.****",
            ],
            
            # 极简输出结构
            "Output_Structure": {
                "description": "BEFORE writing code, you must output a brief Markdown analysis following these 4 sections, followed by the complete code:",
                "sections": [
                    "1. Strategy Summary: How you plan to solve the problem using the provided Blueprint.",
                    "2. Data Pipeline: Key steps for preprocessing and feature engineering.",
                    "3. Model & Training: Architecture, optimizer, and validation setup.",
                    "4. Complete Implementation: A single, fully self-contained Python code block implementing the entire pipeline (train and inference)."
                    
                ]
            },
            "Instructions":{},
        }

# 如果你的框架依赖于特定的解析格式，可以继续合并这一行
        prompt["Output_Structure"] |= self._prompt_resp_fmt

        prompt["Instructions"] |= self._prompt_impl_guideline
        prompt["Instructions"] |= self._prompt_environment

        if self.acfg.data_preview:
            prompt["Data Overview"] = self.data_preview

        plan1, code1 = self.plan_and_code_query(prompt)
        code = code1
        prompt_patch: dict = {
            "Patch Guide": self.scot,
            "Kaggle Blueprint": self.kaggle_tool,
            "baseline code": code1,
        }
        plan2, code2 = self.plan_and_code_query(prompt_patch)
        
        # ==========================================
        # DIFF 模块：解析并应用 Search/Replace (或跳过)
        # ==========================================
        diff_patch_str = ""
        # 使用正则匹配 Search/Replace 块
        blocks = re.findall(r'<<<<<<<\s*SEARCH\n(.*?)\n=======\n(.*?)\n>>>>>>>\s*REPLACE', code2, flags=re.DOTALL)
        new_code = code1
        
        for search_block, replace_block in blocks:
            if search_block in new_code:
                new_code = new_code.replace(search_block, replace_block)
            else:
                logger.warning(f"Could not apply diff block: {search_block[:50]}...")
                
        # 如果成功解析出块，则更新 code 为打补丁后的完整代码
        if blocks:
            code = new_code
        node= Node(plan=plan1+plan2, code=code, parent=None)
        node.root = node.id
        node.node_type = "draft"
        is_duplicated = self.global_memo.process_new_node(node)
        return node
        
    

    def _improve(self, parent_node: Node, ablation: bool = False) -> Node:
            import random # 确保引入 random
            
            # 获取包含 "Persistent Memory(MD)" 和 "Recent Buffer(JSON)" 的融合记忆
            improve_memory = self.buffer_memo.get_memory_context("improve")
            
            # 【修改点 1】：随机决定当前这次尝试是使用 Diff 局部替换，还是 Full Code 全局生成
            use_diff = random.random() < 0.7
            
            if not ablation:
                prompt: Any = {
                    "Resource": {
                        "description": "You can use the following resources",
                        "spec": {
                            "cpu": "5",
                            "gpu": "1",
                            "memory": "40 GB",
                            "gpuType": "nvidia_h100 (80/4GB VRAM)",
                        }
                    },
                    "Task description": self.task_desc,
                    "Strategy Evolution Memory": (
                        "You are equipped with a 'Strategy Evolution Memory' which contains two parts:\n"
                        "1. [Persistent Memory]: A high-level summary of proven synergies (Positive Guidance) and fatal pitfalls (Negative Constraints) from past iterations.\n"
                        "2. [Recent Buffer]: A JSON log of the most recent micro-experiments and their immediate metrics.\n\n"
                        "CRITICAL INSTRUCTIONS ON USING THIS MEMORY:\n"
                        "- AVOID BLIND SPOTS: You MUST NOT propose any method listed in 'Negative Constraints' or repeat any failed plan from the 'Recent Buffer'.\n"
                        "- LEVERAGE SYNERGY: Build your new strategy upon the 'Positive Guidance'. If A and B work well together, keep them and add C.\n"
                        "- INNOVATE: Your proposed improvement MUST be conceptually NOVEL compared to past attempts. Do not just blindly repeat a recent successful action; take the logical next step.\n\n"
                        "=== MEMORY CONTENT ===\n"
                        f"{improve_memory}\n"
                        "======================"
                    ),
                    "Previous solution": {
                        "Code": wrap_code(parent_node.code),
                        "Analysis": parent_node.analysis,
                        "Metric": str(parent_node.metric.value),
                        "Parent Metric": str(parent_node.parent_improve_metric),
                    },
                    "Instructions": {}
                }
    
                # 【修改点 2】：根据 use_diff 动态拼接 Introduction
                base_intro = (
                    "You are a Kaggle grandmaster attending a competition. You are provided with a previously developed "
                    "solution below and should improve it in order to further increase the (test time) performance.\n"
                    "- The submission.csv format **must** fit the sample_submission, but you MUST generate the submission.csv file according to the input. You cannot copy the sample_submission.csv.\n"
                    "- You **MUST NOT** change the number of rows of the original test input.\n"
                    "- Resource limitations **MUST** be strictly followed.\n"
                    "- YOU **MUST** CONSIDER USING PRETRAINED MODELS FROM HUGGING-FACE MIRROR FIRST INSTEAD OF BASIC MACHINE LEARNING TRICKS.\n"
                    "- The runtime environment has no direct access to huggingface.co. Always configure Hugging Face mirror via the HF_ENDPOINT environment variable.\n"
                    "For this task, you should first outline a brief plan in natural language for how the solution can be improved, "
                )
                
                if use_diff:
                    prompt["Introduction"] = base_intro + "and then implement this improvement in Python using a Search/Replace DIFF format based on the provided previous solution."
                else:
                    prompt["Introduction"] = base_intro + "and then implement this improvement in Python by providing the COMPLETE and FULL updated code."
    
                # 【修改点 3】：根据 use_diff 动态注入 Response format 指令
                instructions_dict = self._prompt_resp_fmt.copy()
                if use_diff:
                    instructions_dict["Response format"] = (
                        "****CRITICAL: Your response must be an analysis of strictly 3-5 sentences, "
                        "followed by a single markdown code block containing Search/Replace blocks:\n"
                        "<<<<<<< SEARCH\n[exact code to replace]\n=======\n[new code]\n>>>>>>> REPLACE\n"
                        "DO NOT output full code. Use multiple blocks if needed."
                    )
                else:
                    instructions_dict["Response format"] = (
                        "****CRITICAL: Your response must be an analysis of strictly 3-5 sentences, "
                        "followed by a single markdown code block containing the FULL, complete, and runnable updated Python script.\n"
                        "DO NOT use snippets, placeholders, or diffs. You MUST output the entire rewritten code."
                    )
                prompt["Instructions"] |= instructions_dict
                
                prompt["Instructions"] |= {
                    "Solution improvement sketch guideline": [
                        "The solution sketch should be a brief natural language description of how the previous solution can be improved.",
                        "ATOMIC INCREMENTAL: You should be very specific and propose a SINGLE ACTIONABLE ATOMIC improvement (e.g., adding one specific augmentation, tweaking one specific hyperparameter, modifying one block of the architecture).",
                        "This improvement should be atomic so that we can experimentally evaluate the exact effect of the proposed change.",
                        "MEMORY-DRIVEN NOVELTY: To prove you have read the 'Strategy Evolution Memory', briefly mention WHY this new approach avoids past pitfalls or exploits past synergies.",
                        "- It is **crucial** that your proposed solution **is distinctly different from** the existing designs in the Memory section, yet logically follows the 'Next Exploration Steps'.",
                        "The solution sketch should be strictly 3-5 sentences.",
                        "Don't suggest to do EDA.",
                    ],
                }
                prompt["Instructions"] |= self._prompt_impl_guideline
                
                is_duplicated = True
                for i in range(3):
                    plan, code = self.plan_and_code_query(prompt)
                    
                    # ==========================================
                    # DIFF 模块：解析并应用 Search/Replace (或跳过)
                    # ==========================================
                    diff_patch_str = ""
                    if use_diff:
                        diff_patch_str = code
                        # 使用正则匹配 Search/Replace 块
                        blocks = re.findall(r'<<<<<<<\s*SEARCH\n(.*?)\n=======\n(.*?)\n>>>>>>>\s*REPLACE', code, flags=re.DOTALL)
                        new_code = parent_node.code
                        
                        for search_block, replace_block in blocks:
                            if search_block in new_code:
                                new_code = new_code.replace(search_block, replace_block)
                            else:
                                logger.warning(f"Could not apply diff block: {search_block[:50]}...")
                                
                        # 如果成功解析出块，则更新 code 为打补丁后的完整代码
                        if blocks:
                            code = new_code
                    else:
                        # 如果不是 diff 模式，大模型返回的 code 就是全文，不需要处理，直接使用即可
                        diff_patch_str = "Full code replacement generated."
                    
                    # 创建新节点
                    node = Node(
                        plan=plan,
                        code=code,
                        diff_patch=diff_patch_str,  
                        parent=parent_node,
                        root=parent_node.root,
                        node_type="improve",
                        parent_improve_metric=parent_node.metric.value if hasattr(parent_node, 'metric') and parent_node.metric else None
                    )
                    
                    if self.global_memo.process_new_node(node):
                        is_duplicated = False
                        break
                        
                if is_duplicated:
                    node = Node(
                        plan="[System] Path truncated: LLM exhausted current logical directions and repeatedly generated duplicate code.",
                        code="",  # 死胡同不需要执行代码
                        diff_patch="",  
                        parent=parent_node,
                        root=parent_node.root,
                        bug_mem="",
                        node_type="deadend",
                        parent_improve_metric=parent_node.metric.value if hasattr(parent_node, 'metric') and parent_node.metric else None
                    )
                    parent_node.children.add(node)
                    return node
                    
                parent_node.children.add(node)
                return node
    def _debug(self, parent_node: Node, ablation: bool = False) -> Node:
        current_debug_depth = getattr(parent_node, "debug_depth", 0) + 1
        use_diff = current_debug_depth <= 3
    # 1. 检索相似节点
        retriever = NodeRetriever(self.cfg , self.journal.nodes)
        # 增加 top_k 数量或保持 3，视 Prompt 长度而定
        similar_past_nodes = retriever.search(parent_node, top_k=3)
        similar_ids = [node.id for node in similar_past_nodes] if similar_past_nodes else []
        logger.info(f"similar_past_node id: {similar_ids}")
        # 2. 准备执行输出与记忆
        term_out = self.term_out_cut(parent_node)
        # 3. 生成记忆总结（针对死胡同和逻辑循环）
        debug_memory = self.buffer_memo.get_memory_context("debug")

        # 4. 构造 RAG 参考内容
        reference_str = ""
        for i, node in enumerate(similar_past_nodes):
            # 排除掉当前节点本身
            if node.id == parent_node.id: continue
            reference_str += f"### Reference {i+1} (Score: {node.metric if node.metric else 'N/A'})\n"
            reference_str += f"- Past Bug: {node.exc_type}\n"
            if node.metric:
                reference_str += f"- Successful Solution: {node.plan}\n"
            else :
                reference_str += f"- Failed Solution: {node.plan}\n"
            reference_str += f"- Key Code Snippet:\n{wrap_code(node.code)}\n\n"

        if not ablation:
            prompt: Any = {
                "Resource": {
                    "description": "System hardware constraints",
                    "spec": {
                        "cpu": "5",
                        "gpu": "1",
                        "memory": "40 GB",
                        "gpuType": "nvidia_h100 (80/4GB VRAM)",
                    }
                },

                "Introduction": (
                    "You are a Kaggle Grandmaster tasked with fixing a broken pipeline.\n"
                    "### STRATEGY HIERARCHY ###\n"
                    "1. **Reference-Led Fix**: Priority 1 is to see if a similar bug was solved in 'Historical References'.\n"
                    "2. **Memory-Guided Avoidance**: Use 'Debug Memory' to ensure you don't repeat failed attempts.\n"
                    "3. **Surgical Precision**: Only modify the parts of 'Debug Code' that directly cause the 'Execution bugs'."
                ),

                "Context": {
                    "Task Description": self.task_desc,
                    "Current Buggy Code": wrap_code(parent_node.code),
                    "Execution Error Log": wrap_code(term_out, lang=""),
                },

                "Debug Memory": {
                    "description": "Crucial accumulated knowledge. Contains persistent rules to follow and recent attempts to strictly avoid or successful attempts to learn.",
                    "content": debug_memory
                },

                "Historical References (RAG)": {
                    "description": "Below are 2-3 similar past bugs or familiar successful solution. Use these as templates for your fix.",
                    "content": reference_str if reference_str else "No direct similarities found in past history."
                },

                "Instructions": {
                    "Response Protocol": [
                        "Step 1: Failure Reflection - Explicitly list what failed in 'Debug Memory' and why you MUST NOT do it again.",
                        "Step 2: Root Cause Analysis - Analyze the NEW error log provided and explain exactly why the current code failed.",
                        "Step 3: Analogy - Did any 'Historical Reference' provide a direct fix? If not, why?",
                        "Step 4: Implementation Plan - Explain your surgical fix in 3 sentences.",
                        "Step 5: Diff Patch - Return ONLY exact Search/Replace blocks needed to fix the bug (wrapped in ```python ... ```). Use format:\n<<<<<<< SEARCH\n[exact original code chunk]\n=======\n[new code chunk]\n>>>>>>> REPLACE\nDo NOT output the full code. Be surgical." if use_diff else "Step 6: Final Code - Provide the complete, fully updated python code properly wrapped in ```python ... ``` block."
                    ],
                    "Constraints": [
                        "Never suggest EDA.",
                        "Always configure HF_ENDPOINT for Hugging Face.",
                        "If the error is a 'Dead End' from Memory, you MUST propose a radically different logical path."
                    ]
                }
            }
            instructions_dict = self._prompt_resp_fmt.copy()
            if use_diff:
                instructions_dict["Response format"] = (
                    "****CRITICAL: Your response must follow the 5-step protocol above, "
                    "followed by EXACTLY ONE single markdown code block (```python ... ```) containing all your modifications.\n"
                    "Inside this SINGLE markdown block, you may put multiple Search/Replace blocks if needed, formatted exactly like this:\n"
                    "<<<<<<< SEARCH\n[exact code to replace]\n=======\n[new code]\n>>>>>>> REPLACE\n"
                    "WARNING: DO NOT output multiple markdown blocks. DO NOT output the full code."
                )
            prompt["Instructions"] |= instructions_dict
            prompt["Instructions"] |= self._prompt_impl_guideline

            if self.acfg.data_preview:
                prompt["Data Overview"] = self.data_preview
                
            is_duplicated = True
            for i in range(3):
                plan, code = self.plan_and_code_query(prompt)
                diff_patch_str = ""
                if use_diff:
                    diff_patch_str = code
                    # Simple SEARCH/REPLACE applier
                    blocks = re.findall(r'<<<<<<<\s*SEARCH\n(.*?)\n=======\n(.*?)\n>>>>>>>\s*REPLACE', code, flags=re.DOTALL)
                    new_code = parent_node.code
                    for search_block, replace_block in blocks:
                        if search_block in new_code:
                            new_code = new_code.replace(search_block, replace_block)
                        else:
                            logger.warning(f"Could not apply diff block: {search_block[:50]}...")
                    if blocks:
                        code = new_code
                    # If no blocks found, maybe the LLM outputted full code anyway. Keep code as is.

                node = Node(
                    plan=plan,
                    code=code,
                    diff_patch=diff_patch_str,
                    parent=parent_node,
                    root = parent_node.root,
                    node_type = "debug",
                    parent_improve_metric = parent_node.parent_improve_metric
                )
                if self.global_memo.process_new_node(node):
                    is_duplicated = False
                    break
            if is_duplicated:
                node = Node(
                    plan=plan,
                    code=code,
                    diff_patch=diff_patch_str if use_diff else "",
                    parent=parent_node,
                    root = parent_node.root,
                    node_type = "deadend",
                    parent_improve_metric = parent_node.parent_improve_metric
                )
            parent_node.children.add(node)
            return node

            
    def _explode(self, parent_node: Node, amplitude: float, num_sparks: int, forced_temp: float | None = None) -> list[Node]:
            """
            FWA Explosion: 根据半径(amplitude)产生多个火花节点（并行生成）。
            amplitude 接近 0: 精英节点，进行 Fine-tuning (低 Temp, 局部改动)
            amplitude 接近 1: 差节点，进行 Radical Mutation (高 Temp, 结构改动)
            """
            from concurrent.futures import ThreadPoolExecutor, as_completed
            import threading
            import random

            sparks = []
            improve_memory = self.buffer_memo.get_memory_context("improve")
            sparks_lock = threading.Lock()

            # 1. 根据幅度精细化指令逻辑，并强绑定 use_diff 策略
            # forced_temp 优先级最高：当强制爆炸时，直接覆盖温度设置
            if forced_temp is not None:
                temp = forced_temp
                use_diff = False  # forced explode always uses full code for maximum diversity
                severity_desc = (
                    f"FOCUS: FORCED RADICAL BREAKOUT (Temperature={temp}). The search has stagnated. "
                    "You MUST make FUNDAMENTAL changes: switch model backbone, redesign data pipeline, "
                    "or try a completely different approach. DO NOT make incremental tweaks."
                )
            elif amplitude < 0.3:
                severity_desc = (
                    "FOCUS: PRECISION TUNING. You are in a 'local optimum' search. "
                    "Modify only small components (e.g., Weight Decay, Learning Rate, slight architecture tweaks). "
                    "KEEP the core logic 95% identical."
                )
                temp = 1.0
                use_diff = True
            elif amplitude > 0.7:
                severity_desc = (
                    "FOCUS: RADICAL RECONSTRUCTION. The current approach is plateauing. "
                    "CHANGE the core backbone (e.g., CNN -> Transformer), or rewrite the data processing pipeline. "
                    "Be bold and distinctly different from the provided solution."
                )
                temp = 1.0
                use_diff = False
            else:
                severity_desc = "FOCUS: BALANCED EVOLUTION. Improve features and refine model integration."
                temp = 1.0
                use_diff = random.choice([True, False])

            # 2. 为每个火花预构建独立的 prompt（每个火花有不同的 diversity angle）
            diversity_angles = [
                "Focus on data augmentation and preprocessing changes.",
                "Focus on model architecture modifications (layers, heads, activations).",
                "Focus on training strategy (optimizer, scheduler, regularization).",
                "Focus on feature engineering and representation learning.",
                "Focus on ensemble methods and post-processing techniques.",
                "Focus on loss function and metric optimization.",
                "Focus on hyperparameter search and automated tuning.",
                "Focus on transfer learning and pretrained model adaptation.",
            ]

            def build_spark_prompt(spark_idx: int) -> dict:
                """为单个火花构建 prompt"""
                # 为每个火花分配一个独特的 diversity angle
                angle = diversity_angles[spark_idx % len(diversity_angles)]

                prompt: Any = {
                    "Resource": {
                        "spec": {
                            "cpu": "5",
                            "gpu": "1",
                            "memory": "40 GB",
                            "gpuType": "nvidia_h100 (80/4GB VRAM)",
                        }
                    },
                    "Introduction": (
                        f"You are a Kaggle Grandmaster performing an **Explosion Search (Spark {spark_idx+1}/{num_sparks})**. "
                        f"The target mutation intensity is {amplitude:.2f}. {severity_desc}\n"
                        "For this task, first outline a brief plan, and then implement it. "
                        f"The implementation MUST use {'Search/Replace DIFF format' if use_diff else 'FULL complete code'}."
                    ),
                    "Task description": self.task_desc,
                    "Strategy Evolution Memory": (
                        "You are equipped with a 'Strategy Evolution Memory' which contains two parts:\n"
                        "1. [Persistent Memory]: A high-level summary of proven synergies and fatal pitfalls.\n"
                        "2. [Recent Buffer]: A JSON log of recent micro-experiments.\n\n"
                        "CRITICAL INSTRUCTIONS ON USING THIS MEMORY:\n"
                        "- AVOID BLIND SPOTS: You MUST NOT propose any method listed in 'Negative Constraints'.\n"
                        "- LEVERAGE SYNERGY: Build your new strategy upon the 'Positive Guidance'.\n"
                        "- INNOVATE: Your proposed improvement MUST be conceptually NOVEL.\n\n"
                        "=== MEMORY CONTENT ===\n"
                        f"{improve_memory}\n"
                        "======================"
                    ),
                    "Previous solution": {
                        "Code": wrap_code(parent_node.code),
                        "Analysis": parent_node.analysis,
                        "Metric": str(parent_node.metric.value),
                        "Parent Metric": str(parent_node.parent_improve_metric),
                    },
                    "Instructions": {}
                }

                # 设置返回格式
                instructions_dict = self._prompt_resp_fmt.copy()
                if use_diff:
                    instructions_dict["Response format"] = (
                        "****CRITICAL: Your response must be an analysis of strictly 3-5 sentences, "
                        "followed by EXACTLY ONE single markdown code block (```python ... ```) containing all your modifications.\n"
                        "Inside this SINGLE markdown block, you may put multiple Search/Replace blocks if needed, formatted exactly like this:\n"
                        "<<<<<<< SEARCH\n[exact code to replace]\n=======\n[new code]\n>>>>>>> REPLACE\n"
                        "WARNING: DO NOT output multiple markdown blocks. DO NOT output the full code."
                    )
                else:
                    instructions_dict["Response format"] = (
                        "****CRITICAL: Your response must be an analysis of strictly 3-5 sentences, "
                        "followed by a single markdown code block containing the FULL, complete, and runnable updated Python script.\n"
                        "DO NOT use snippets, placeholders, or diffs. You MUST output the entire rewritten code."
                    )
                prompt["Instructions"] |= instructions_dict

                # 爆炸特有的引导：每个火花有独特的 diversity angle
                explosion_guidelines = [
                    f"Current Spark ID: {spark_idx+1}.",
                    f"**DIVERSITY ANGLE**: Your approach should lean towards: {angle}",
                    "DO NOT suggest EDA. Propose one atomic, actionable change.",
                    "If using Pretrained models, always use HF_ENDPOINT for the mirror.",
                    f"Mutation Severity: {severity_desc}",
                    "CRITICAL: Other sparks are being generated IN PARALLEL. You MUST produce a distinctly different approach from typical solutions."
                ]
                prompt["Instructions"] |= {"Explosion-Specific Guidelines": explosion_guidelines}
                prompt["Instructions"] |= self._prompt_impl_guideline

                return prompt

            # 3. 单个火花的生成任务（线程安全）
            def generate_single_spark(spark_idx: int) -> Node:
                """生成单个火花（可在线程池中并行执行）"""
                prompt = build_spark_prompt(spark_idx)

                is_duplicated = True
                plan, code = "", ""
                for attempt in range(3):
                    plan, code = self.plan_and_code_query_explode(prompt, temperature=temp)

                    # DIFF 解析模块
                    diff_patch_str = ""
                    if use_diff:
                        diff_patch_str = code
                        blocks = re.findall(r'<<<<<<<\s*SEARCH\n(.*?)\n=======\n(.*?)\n>>>>>>>\s*REPLACE', code, flags=re.DOTALL)
                        new_code = parent_node.code
                        for search_block, replace_block in blocks:
                            if search_block in new_code:
                                new_code = new_code.replace(search_block, replace_block)
                            else:
                                logger.warning(f"Spark {spark_idx+1}: Could not apply diff block: {search_block[:50]}...")
                        if blocks:
                            code = new_code
                    else:
                        diff_patch_str = "Full code replacement generated for Radical Mutation."

                    node = Node(
                        plan=plan,
                        code=code,
                        diff_patch=diff_patch_str,
                        parent=parent_node,
                        root=parent_node.root,
                        node_type="improve",
                        parent_improve_metric=parent_node.metric.value if hasattr(parent_node, 'metric') and parent_node.metric else None
                    )

                    # 去重检查（global_memo 内部有锁，线程安全）
                    if self.global_memo.process_new_node(node):
                        is_duplicated = False
                        break

                if is_duplicated:
                    node = Node(
                        plan="[System] Path truncated: LLM exhausted current logical directions and repeatedly generated duplicate code.",
                        code="",
                        diff_patch="",
                        parent=parent_node,
                        root=parent_node.root,
                        node_type="deadend",
                        parent_improve_metric=parent_node.metric.value if hasattr(parent_node, 'metric') and parent_node.metric else None
                    )

                logger.info(f"Spark {spark_idx + 1}/{num_sparks} generated (Plan: {node.plan[:50]}...)")
                return node

            # 4. 并行生成所有火花
            max_workers = min(num_sparks, 8)  # 最多 8 个并发
            logger.info(f"[Parallel Explosion] Generating {num_sparks} sparks with {max_workers} workers (amplitude={amplitude:.2f}, temp={temp})")

            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_to_idx = {
                    executor.submit(generate_single_spark, idx): idx
                    for idx in range(num_sparks)
                }
                for future in as_completed(future_to_idx):
                    spark_idx = future_to_idx[future]
                    try:
                        node = future.result()
                        with sparks_lock:
                            sparks.append(node)
                            parent_node.children.add(node)
                    except Exception as e:
                        logger.error(f"Spark {spark_idx + 1} failed with exception: {e}")
                        # 生成一个 deadend 节点作为占位
                        dead_node = Node(
                            plan=f"[System] Spark generation failed: {e}",
                            code="",
                            diff_patch="",
                            parent=parent_node,
                            root=parent_node.root,
                            node_type="deadend",
                            parent_improve_metric=parent_node.metric.value if hasattr(parent_node, 'metric') and parent_node.metric else None
                        )
                        with sparks_lock:
                            sparks.append(dead_node)
                            parent_node.children.add(dead_node)

            # 按 spark_idx 排序以保持顺序一致性（可选，便于日志阅读）
            # sparks 已经通过 as_completed 收集，顺序不确定，但不影响功能
            logger.info(f"[Parallel Explosion] All {len(sparks)} sparks generated.")
            return sparks



    
    # def _check(self, node):
    #     plan = node.plan
    #     code = node.code
        
    #     # 1. 识别是否为 draft 节点
    #     is_draft = getattr(node, "node_type", "") == "draft"

    #     # [基础底线]：无论怎么改，都必须遵守的算力和防爆显存底线
    #     sanity_guidelines = [
    #         "1. [BUG_SANITY] Check syntax, missing imports (torch, pandas, etc.), and device mismatches. Ensure all tensors are handled on H100.",
    #         "2. [DATALOADER_SHM_SAFETY] CRITICAL: Set num_workers=0 and pin_memory=False in DataLoaders to prevent shared memory OOM.",
    #         "3. [PROCESS_ANALYSIS] Compute and log the 'Generalization Audit' (train vs val metric gap) and provide actionable insights.",
    #         "4. [PREDICTION_INDEPENDENCE] Inference MUST be independent. Reload saved weights to run inference.",
    #         "5. [HUGGING_FACE HUB] CRITICAL: Always use os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com' for model downloads.",
    #         "6. [VRAM_EFFICIENCY] MANDATORY: For large models, use Gradient Accumulation (steps >= 4) and Gradient Checkpointing. Batch Size 4-16. Explicitly add `torch.cuda.empty_cache()`."
    #     ]

    #     # 强制使用 Diff 模式进行外科手术式修复
    #     use_diff = True 
    #     instructions_dict = {
    #         "Mandatory Sanity Checklist": sanity_guidelines,
    #     }

    #     # 2. 核心重构：专治 Draft 节点的“方案降级”
    #     if is_draft:
    #         instructions_dict["🚨 BLUEPRINT ANTI-DEGRADATION PROTOCOL (CRITICAL)"] = [
    #             "LLMs frequently suffer from 'Solution Degradation', where they simplify complex multi-stage Blueprints into basic baselines to save tokens. YOU MUST PREVENT THIS.",
    #             "1. [STRUCTURAL ALIGNMENT]: Rigorously map the 'Proposed Code' against the 'Blueprint'. If the Blueprint mandates a 2-stage pipeline (e.g., 3D Segmentation -> 2.5D Sequence Classification), and the code only has a basic 2D baseline, the code is DEGRADED.",
    #             "2. [MISSING PIPELINE INJECTION]: You MUST inject missing Blueprint components (e.g., new Datasets for 2.5D slicing, LSTM Model classes, Mask Injection logic). Use Search/Replace to append these missing classes to the file.",
    #             "3. [PRESERVE DIVERSITY via GRAFTING]: Do NOT rewrite the entire script. Preserve the LLM's original basic classes, stylistic choices, and augmentations. Instead, GRAFT the Blueprint logic into the existing structure (e.g., rewrite `main()` to call the new Blueprint pipelines while keeping the original utility functions intact).",
    #             "4. [EXEMPTION RULE]: You may ONLY ignore a Blueprint trick if the Proposed Plan explicitly provides a superior ML/Scientific justification. 'Reducing complexity' or 'imagined resource limits' are INVALID. If unjustified, FORCE-INJECT the Blueprint."
    #         ]
            
    #         intro_text = (
    #             "You are an Elite Kaggle Competition Auditor. Your task is to review a proposed plan and implementation against the official solution Blueprint. "
    #             "Your primary goal is **STRUCTURAL COMPLETENESS**. If the code is missing key Blueprint architectures (e.g., multi-stage processing, sequence heads), you MUST inject them using large Diff patches. "
    #             "For minor bugs, use Minimal Intervention."
    #         )
    #     else:
    #         instructions_dict["Implementation Audit"] = [
    #             "Ensure <CODE> fully implements ALL items in <PLAN>."
    #         ]
    #         intro_text = (
    #             "You are an Elite Kaggle Competition Auditor. Your task is to review code implementation. "
    #             "Your primary goal is **Minimal Intervention**: DO NOT rewrite code that already satisfies the guidelines. Only modify violating lines."
    #         )

    #     instructions_dict["Modification Principles"] = [
    #         "1. STRICT ADHERENCE: If a section of code is bug-free and compliant, preserve it.",
    #         "2. SURGICAL BUGS: For inline memory tricks or API fixes, use minimal Search/Replace.",
    #         "3. STRUCTURAL GRAFTING (For Drafts): If entire Blueprint stages are missing, use an 'Anchor Diff' to append them. Find a safe anchor (like the end of an existing class), and replace it with the anchor PLUS the new missing classes."
    #     ]

    #     instructions_dict["Solution improvement sketch guideline"] = [
    #         "The refined plan should be a natural language description detailing the exact Blueprint pipelines restored and the bugs fixed.",
    #     ]

    #     # 3. Diff 输出格式（保持原样，强调可以使用大段 Diff）
    #     if use_diff:
    #         instructions_dict["Implementation guideline"] = [
    #             "DO NOT output the full Python script.",
    #             "Ensure the SEARCH block perfectly matches the existing original code character-by-character.",
    #             "You can use large REPLACE blocks to inject entirely new missing classes or massive structural changes."
    #         ]
    #         instructions_dict["Response Format"] = (
    #             "You MUST output your response strictly in the following XML-tag format:\n"
    #             "<REPORT>\n[Detail Blueprint alignments made and bugs fixed]\n</REPORT>\n"
    #             "<PLAN>\n[The refined plan]\n</PLAN>\n"
    #             "<CODE>\n"
    #             "****CRITICAL: Inside this tag, provide EXACTLY ONE single markdown code block (```python ... ```).\n"
    #             "Inside this block, put multiple Search/Replace blocks:\n"
    #             "<<<<<<< SEARCH\n[exact code to replace (or anchor point to append after)]\n=======\n[new code / grafted logic]\n>>>>>>> REPLACE\n"
    #             "WARNING: DO NOT output multiple markdown blocks. DO NOT output the full code.\n"
    #             "</CODE>"
    #         )
    #     else:
    #         # Fallback (保持原样)
    #         instructions_dict["Implementation guideline"] = [
    #             "The final code MUST be a complete, ready-to-run script.",
    #             "CRITICAL: Set num_workers=0 in all DataLoaders.",
    #             "Do not use markdown code blocks inside the XML tags."
    #         ]
    #         instructions_dict["Response Format"] = (
    #             "You MUST output your response strictly in the following XML-tag format:\n"
    #             "<REPORT>\n[Detail only the specific items that were fixed. If none, state 'No changes required']\n</REPORT>\n"
    #             "<PLAN>\n[The refined plan (original plan + specific modifications)]\n</PLAN>\n"
    #             "<CODE>\n[The final Python code (original code + surgical fixes)]\n</CODE>"
    #         )
            
    #     audit_target = {
    #         "Proposed Plan": plan,
    #         "Proposed Code": code,
    #     }
    #     if is_draft:
    #         audit_target["Blueprint"] = self.kaggle_tool

    #     audit_prompt = {
    #         "Resource": {
    #             "description": "The auditing environment is a High-Performance Compute Slot (Shared H100).",
    #             "spec": {
    #                 "cpu_cores_allocated": "5",
    #                 "gpu_vram_quota": "20 GB (of 80GB H100)",
    #                 "memory_ram": "32 GB",
    #                 "optimization_target": "Maximize model performance within VRAM quota using AMP and Gradient Accumulation."
    #             }
    #         },
    #         "Introduction": intro_text,
    #         "Task description":self.task_desc,
    #         "Audit Target": audit_target,
    #         "Instructions": instructions_dict
    #     }
    #     response_text = None
        
    #     for i in range(3):
    #         raw_response = query(
    #             system_message=audit_prompt,
    #             user_message=None,
    #             temperature=self.acfg.draft.temp,
    #             model=self.acfg.draft.model,
    #             cfg=self.cfg
    #         )
    #         # --- 鲁棒性检查开始 ---
    #         if not raw_response:
    #             logger.warning(f"Attempt {i+1}: Received empty response, retrying...")
    #             continue
            
    #         valid_text = None
    #         if isinstance(raw_response, (list, tuple)):
    #             if len(raw_response) > 0 and raw_response[0]:
    #                 valid_text = raw_response[0]
    #         elif isinstance(raw_response, str):
    #             valid_text = raw_response

    #         if valid_text:
    #             response_text = valid_text
    #             logger.info(f"Audit query succeeded on attempt {i+1}")
    #             break
    #         # --- 鲁棒性检查结束 ---
            
    #         logger.warning(f"Attempt {i+1} failed, retrying...")

    #     if not response_text:
    #         logger.error("All audit attempts failed or returned malformed data.")
    #         return Node(
    #             plan=plan,
    #             code=code,
    #             report="Audit Error: Missing or malformed response from LLM.",
    #         )

    #     def extract_tag(tag: str, text: str) -> str | None:
    #         pattern = f"<{tag}>\s*(.*?)\s*</\s*{tag}\s*>"
    #         match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
    #         if match:
    #             content = match.group(1).strip()
    #             if tag.upper() == "CODE":
    #                 content = re.sub(r'^\s*```[a-zA-Z0-9]*\s*\n', '', content)
    #                 content = re.sub(r'\n\s*```\s*$', '', content)
    #             return content.strip()
    #         return None

    #     report = extract_tag("REPORT", response_text)
    #     refined_plan = extract_tag("PLAN", response_text)
    #     extracted_code_str = extract_tag("CODE", response_text)

    #     # 失败回退机制
    #     if not refined_plan or report is None:
    #         try:
    #             logger.error("Critical: Audit Agent failed to return mandatory tags.")
    #         except NameError:
    #             print("Critical: Audit Agent failed to return mandatory tags.")

    #         return Node(
    #             plan=plan,
    #             code=code,
    #             report="Audit Error: Tags missing from LLM response.",
    #         )

    #     # 4. DIFF 解析模块：将补丁应用到原始代码
    #     final_refined_code = code # 默认继承原代码
        
    #     if use_diff and extracted_code_str:
    #         # 匹配所有的 SEARCH / REPLACE 块
    #         blocks = re.findall(r'<<<<<<<\s*SEARCH\n(.*?)\n=======\n(.*?)\n>>>>>>>\s*REPLACE', extracted_code_str, flags=re.DOTALL)
            
    #         if blocks:
    #             for search_block, replace_block in blocks:
    #                 if search_block in final_refined_code:
    #                     final_refined_code = final_refined_code.replace(search_block, replace_block)
    #                 else:
    #                     # 降级处理：如果因为空格或缩进导致精确匹配失败，这里可以记录日志提醒
    #                     try:
    #                         logger.warning(f"Audit Diff patch apply failed: could not find exact block matching {search_block[:30]}...")
    #                     except NameError:
    #                         print(f"Warning: Diff patch apply failed for block {search_block[:30]}")
    #         else:
    #             # 兼容 LLM 没有严格遵守 DIFF 格式，但输出了完整代码的情况
    #             if len(extracted_code_str) > 200 and "def " in extracted_code_str:
    #                 final_refined_code = extracted_code_str
    #     elif not use_diff and extracted_code_str:
    #         final_refined_code = extracted_code_str

    #     return Node(
    #         plan=refined_plan,
    #         code=final_refined_code,
    #         report=report,
    #     )
    
    def _merge(self, parent_node1: Node, parent_node2: Node) -> Node:
        # 提取两者的关键元数据
        p1_summary = {
            "score": str(parent_node1.metric.value),  # 假设 Node 有 score 属性
            "last_node_metric": str(parent_node1.parent_improve_metric),
            "analysis": parent_node1.analysis,
            "plan": parent_node1.plan,
            "code": wrap_code(parent_node1.code)
        }
        p2_summary = {
            "score": str(parent_node2.metric.value),
            "last_node_metric": str(parent_node1.parent_improve_metric),
            "analysis": parent_node2.analysis,
            "plan": parent_node2.plan,
            "code": wrap_code(parent_node2.code)
        }

        logger.info(f"Start Vote+Evolve Fusion: Parent1({parent_node1.id}) vs Parent2({parent_node2.id})")

        prompt: Any = {
            "Resource": {
                "spec": {
                        "cpu": "5",
                        "gpu": "1",
                        "memory": "40 GB",
                        "gpuType": "nvidia_h100 (80/4GB VRAM)",    
                }
            },
            "Task description": self.task_desc,
            "Parent_1": p1_summary,
            "Parent_2": p2_summary,
            "Instructions": {}
        }

        # --- Step 1: Evaluation & Voting (主从定标) ---
        prompt["Instructions"] |= {
            "Step 1: Evaluation & Voting": (
                "Critically evaluate Parent 1 (the current best performer) and Parent 2 (selected for its maximum semantic diversity). "
                "1. **Determine the Base**: Select the solution with the most robust logic and scientific validation strategy as the **Base Solution**. "
                "2. **Identify Donor DNA**: From the other solution (the **Donor Solution**), identify 1-2 specific 'High-Value Components' "
                "(e.g., a unique feature engineering trick, a more robust regularization, or a specialized post-processing logic) "
                "that the Base Solution currently lacks. "
                "3. **Diversity Value**: Specifically look for what Parent 2's unique perspective can offer to overcome Parent 1's limitations."
            )
        }
        
        # --- Step 2: Evolutionary Fusion (择优进化) ---
        prompt["Instructions"] |= {
            "Step 2: Evolution Strategy (Cherry-Picking)": (
                "Instead of a blind merge, perform an 'Evolutionary Refinement' to create a superior iteration:\n"
                "1. **Template**: Use the **Base Solution** code as your primary architectural template.\n"
                "2. **Injection**: Integrate the identified 'High-Value Components' from the **Donor Solution** into the template.\n"
                "3. **Conflict Resolution**: If Donor logic contradicts Base logic, prioritize the Base. "
                "**EXCEPTION**: If the Donor's approach clearly addresses a failure or bottleneck documented in the 'Historical Bug Memory', "
                "adopt the Donor's implementation for that specific part.\n"
                "4. **Resource Constraints**: Maintain GPU memory efficiency. Do not redundantly stack models or heavy layers. "
                "5. **Goal**: Ensure the New Plan explains how this 'cross-pollination' of different styles improves the overall solution."
            )
        }

        # --- Step 3: Formatting & Implementation ---
        prompt["Instructions"] |= self._prompt_resp_fmt
        prompt["Instructions"] |= {
            "Step 3: Implementation Guideline": (
                "Generate the final 'Plan' detailing why the base was chosen and what specific 'genes' were taken from the donor. "
                "Then output the 'Code'. Ensure the code remains clean and does not contain redundant logic from the weaker parent."
            )
        }
        prompt["Instructions"] |= self._prompt_impl_guideline

        plan, code = self.plan_and_code_query(prompt)
        
        is_duplicated = True
        for i in range(3):
            plan, code = self.plan_and_code_query(prompt)
            res = Node(
                plan=plan,
                code=code,
                parent=None,  # 记录双亲
                _logical_parents = [parent_node1, parent_node2],
                improve_mem=f"Fused from {parent_node1.id} and {parent_node2.id} using Vote+Evolve.",
                node_type = "merge"
            )
            if self.global_memo.process_new_node(res):
                is_duplicated = False
                break
        res.root = res.id
        
        return res
    
    def update_data_preview(
            self,
    ):
        self.data_preview = data_preview.generate(self.cfg.workspace_dir)

    def step(self,nodes:list[Node],policy:str):
        if not self.journal.nodes or self.data_preview is None:
            self.update_data_preview()
        if not nodes:
            new_nodes = [self._draft()]
            
        else:
            res = (nodes,policy)
            parent_node_list, action = res if isinstance(res, tuple) else (res, "debug")
            new_nodes = []

            if action == "debug":
                new_nodes.append(self._debug(parent_node_list[0], ablation=False))
            elif action == "merge":
                new_nodes.append(self._merge(parent_node_list[0], parent_node_list[1]))
            elif action == "improve":
                new_nodes.append(self._improve(parent_node_list[0], ablation=False))
            elif action == "forced_explode":
                # Forced high-temp explode to break stagnation
                sparks = self._forced_explode()
                new_nodes.extend(sparks)
            elif action == "explode":
                target_node = parent_node_list[0]
                good_nodes = self.journal.good_nodes
                if len(good_nodes) < 2:
                    num_sparks = self.acfg.search.min_sparks
                    amplitude = 0.5
                else:
                    all_metric = [n.metric.value for n in good_nodes]
                    f_min, f_max = np.min(all_metric), np.max(all_metric)
                    f_range = f_max - f_min + 1e-9

                    # --- 核心修改：根据目标方向转换 metric 表现 ---
                    # 我们定义 quality_score：无论原始指标如何，quality_score 越大代表表现越好
                    if target_node.metric.maximize:
                        # 越大越好：直接计算与最小值的差
                        quality_diff = target_node.metric.value - f_min + 1e-9
                        all_quality_diffs = [m - f_min + 1e-9 for m in all_metric]
                        # 半径计算：值越大，距离 f_max 越近，半径越小
                        dist_to_best = f_max - target_node.metric.value
                    else:
                        # 越小越好（Maximize=False）：反转逻辑，计算与最大值的差
                        quality_diff = f_max - target_node.metric.value + 1e-9
                        all_quality_diffs = [f_max - m + 1e-9 for m in all_metric]
                        # 半径计算：值越小，距离 f_min 越近，半径越小
                        dist_to_best = target_node.metric.value - f_min

                    # --- 计算分配比例 ---
                    denom_s = np.sum(all_quality_diffs) + 1e-9

                    # 1. 计算爆炸个数 S_i：表现越好(quality_diff越大)，分配火花越多
                    s_i = self.acfg.search.total_sparks * (quality_diff / denom_s)
                    num_sparks = int(np.clip(s_i, self.acfg.search.min_sparks, self.acfg.search.max_sparks))

                    # 2. 计算爆炸半径 A_i：表现越好(dist_to_best越小)，半径越小
                    # 归一化半径在 [0.2, 0.9] 之间
                    amplitude = dist_to_best / f_range
                    amplitude = 0.2 + 0.7 * amplitude 

                logger.info(f"FWA Exploding: Node {target_node.id}, Sparks: {num_sparks}, Amplitude: {amplitude:.4f}")

                    # 调用爆炸函数
                sparks = self._explode(target_node, amplitude, num_sparks)
                new_nodes.extend(sparks)
        res =[]
        for node in new_nodes:
            # ck_node = self._check(node)
            # if ck_node.plan and ck_node.plan != node.plan:
            #         node.plan = (
            #             f"### Original Plan ###\n{node.plan}\n\n"
            #             f"### Audit Refinement ###\n{ck_node.plan}"
            #         )
            # node.code = ck_node.code
            # node.report = ck_node.report
            res.append(node)
        return res

    def normalize_bool(self, x):
        if isinstance(x, bool):
            return x
        if isinstance(x, str):
            return x.lower() == "true"

    def safe_float(self, x):
        try:
            return float(x)
        except (TypeError, ValueError):
            return None

    def is_better_than(self, node: Node, best: Node | None) -> bool:
        # 没有 best，一定更好
        if best is None:
            return True

        # worst 永远不可能更好
        if node.metric.is_worst:
            return False

        # best 是 worst，node 只要不是 worst 就赢
        if best.metric.is_worst:
            return True

        # maximize 必须一致，否则视为不可比较
        if node.metric.maximize != best.metric.maximize:
            logger.warning(
                f"Inconsistent maximize flag: node={node.metric.maximize}, best={best.metric.maximize}"
            )
            return False

        return node.metric > best.metric

    def term_out_cut(self, node: Node, max_line=100) -> str:
        lines = node._term_out or []
        lines = [l for l in lines if l.strip() != ""]  # 去掉纯空行

        # --- 改进点：关键词保护机制 ---
        # 定义你绝对不想错过的关键词
        critical_keywords = ["Accuracy", "Index", "Gap", "Metric", "Score", "Final"]

        important_lines = []
        # 从后往前找，优先保留带有关键信息的行
        for line in reversed(lines):
            if any(key.lower() in line.lower() for key in critical_keywords):
                important_lines.append(line)
            if len(important_lines) >= 10:  # 最多保留10条关键指标行
                break

        # 取最后的 N 行作为上下文
        tail_lines = lines[-max_line:] if len(lines) > max_line else lines

        # 组合：重要行 + 最后的上下文（去重并保持顺序）
        # 这里用 set 来简单处理，或者直接相信 tail_lines 已经覆盖了大部分情况
        term_out = "\n".join(tail_lines)

        if node.exc_type is not None:
            term_out += f"\n\nException Type:\n{node.exc_type}"
            if node.exc_info:
                term_out += "\nException Info:\n" + json.dumps(node.exc_info, indent=2)
            if node.exc_stack:
                term_out += "\nException Stack:\n" + json.dumps(node.exc_stack, indent=2)

        return term_out

    def parse_exec_result(self, node: Node, exec_result: ExecutionResult, ablation: bool = False):
        # 解析执行结果，更新节点状态(is_bug)，metric
        logger.info(f"Agent is parsing execution results for node {node.id}")

        node.absorb_exec_result(exec_result)
        term_out = self.term_out_cut(node)
        logger.info(f"term_out after cut:{term_out}")
        global_step = len(self.journal)
        response = self.evaluator.evaluate(term_out, node , global_step)
        
        is_bug = self.normalize_bool(response["is_bug"])
        lower_is_better = self.normalize_bool(response["lower_is_better"])
        metric = self.safe_float(response["metric"])
        gpu_required = self.normalize_bool(response["gpu_required"])

        if not isinstance(metric, float):
            response["metric"] = None

        node.analysis = response["summary"]

        node.is_buggy = (
                is_bug
                or metric is None
                or node.exc_type is not None
        )
        # bug节点的metric评价最低
        if node.is_buggy:
            node.metric = WorstMetricValue()
        else:

            node.metric = MetricValue(
                metric, maximize=not lower_is_better
            )
            # node.gap = gap
            if not ablation:
                if self.is_better_than(node, self.best_node if self.best_node else None):
                    # 进行判定，看submission是不是真的更好：
                    snapshot_success = self.evaluator.save_submission_snapshot(node)
                    if snapshot_success:
                        # 只有快照保存成功（文件存在且非空），才认为这是新的“最佳节点”
                        self.best_node = node
                            
                        logger.info(f"🏆 Best node updated: Node {node.id}, Metric: {node.metric.value}")
                    else:
                        # 如果文件不存在，即使分数再高也判定为无效更新，防止“错误更新”
                        logger.warning(f"⚠️ Node {node.id} has better metric ({node.metric.value}), but failed submission check. Ignoring.")
                        
        if node.node_type =='improve' or node.node_type =='debug':
            logger.info(f"Adding node {node.id} ({node.node_type}) to memory buffer.")
            self.buffer_memo.add_node(node)

