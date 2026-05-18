import json
import os
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional ,Union
from aide.journal import Node
from aide.backend import query,FunctionSpec
import re
import logging
import threading
from pathlib import Path
logger = logging.getLogger("aide")
class AgentMemoryManager:
    def __init__(self,cfg , buffer_limit: int = 4):
        self.buffer_limit = buffer_limit
        # 假设 cfg.workspace_dir 是字符串或 Path
        base_dir = Path(cfg.workspace_dir) / "memory"
        
        # 自动创建目录（如果不存在的话），防止后续保存 JSON 时报错
        base_dir.mkdir(parents=True, exist_ok=True)
        
        # 最终的文件路径
        self.improve_json = base_dir/"improve_buffer.json"
        self.debug_json = base_dir/"debug_buffer.json"
        self.improve_md = base_dir/"improve.md"
        self.debug_md = base_dir/"debug.md"
        self.cfg = cfg
        self.acfg = cfg.agent
        self.buffer_lock = threading.Lock()
        self.improve_func_spec = FunctionSpec(
            name="record_improve_summary",
            description="Record a staged experience summary for Model/Code Optimization (Improve) tasks.",
            json_schema={
                "type": "object",
                "properties": {
                    "positive_guidance": {
                        "type": "string",
                        "description": "Effective Optimization Strategies: Extract proven methodologies that led to metric improvement. (CRITICAL: Limit this explanation to 3-5 sentences strictly.)"
                    },
                    "negative_constraints": {
                        "type": "string",
                        "description": "Ineffective/Negative Patterns: Actions that dropped or froze the metric, acting as a clear pitfall warning. (CRITICAL: Limit this explanation to 3-5 sentences strictly.)"
                    },
                    "synergy_observations": {
                        "type": "string",
                        "description": "Potential Synergies: Effects produced by combining different tricks, or strategies that must be used together. (CRITICAL: Limit this explanation to 3-5 sentences strictly.)"
                    },
                    "next_steps": {
                        "type": "string",
                        "description": "Next Exploration Steps: Recommended code directions for the Agent to focus on based on the current context. (CRITICAL: Limit this explanation to 3-5 sentences strictly.)"
                    }
                },
                "required": ["positive_guidance", "negative_constraints", "synergy_observations", "next_steps"]
            }
        )
        # 2. Debug 专属 Schema
        self.debug_func_spec = FunctionSpec(
            name="record_debug_summary",
            description= "Record a staged experience summary for Code Repair (Debug) tasks.",
            json_schema={
                "type": "object",
                "properties": {
                    "frequent_error_patterns": {
                        "type": "string",
                        "description": "Frequent Error Patterns: Identify the root causes of errors (e.g. mismatching dims, GPU OOM, etc). (CRITICAL: Limit this explanation to 3-5 sentences strictly.)"
                    },
                    "standard_fixes": {
                        "type": "string",
                        "description": "Standard Fixes: Summarize the ultimate solutions and the path that resolved the errors. (CRITICAL: Limit this explanation to 3-5 sentences strictly.)"
                    },
                    "defensive_coding_rules": {
                        "type": "string",
                        "description": "Defensive Coding Rules: Propose standards preventing future Agents from writing code prone to similar issues. (CRITICAL: Limit this explanation to 3-5 sentences strictly.)"
                    }
                },
                "required": ["frequent_error_patterns", "standard_fixes", "defensive_coding_rules"]
            }
)
        self._init_files()

    def _init_files(self):
        """初始化缓冲池文件"""
        for file in [self.improve_json, self.debug_json]:
            if not os.path.exists(file):
                with open(file, 'w', encoding='utf-8') as f:
                    json.dump([], f)
                    
    def save_buffer_safely(self, node: Node) -> dict:
        """
        绝对安全的缓冲池序列化方法。
        完全弃用 asdict() / to_dict()，手动提取所需字段，彻底杜绝循环引用。
        """
        # 手动提取交由 LLM 总结的核心字段，避免序列化整个对象
        safe_dict = {
            "id": str(node.id),
            "plan": getattr(node, "plan", ""),
            "diff_patch":getattr(node, "diff_patch", ""),
            "node_type": getattr(node, "node_type", ""),
            "is_buggy": getattr(node, "is_buggy", False),
            "exc_type": getattr(node, "exc_type", ""),
            "analysis": getattr(node, "analysis", ""),
            "report": getattr(node, "report", ""),
            
            # 分数提取，安全处理 None 的情况
            "metric": node.metric.value if (hasattr(node, "metric") and node.metric) else None,
            "parent_improve_metric": getattr(node, "parent_improve_metric", None),
            
            # 【斩断循环的核心】：不要存对象，只存 ID 字符串！
            "parent_id": node.parent.id if getattr(node, "parent", None) else None,
        }
        
        # 对于 Debug 节点，我们需要保留一部分报错信息供大模型分析
        if safe_dict["node_type"] == "debug":
            # 如果之前保存了 term_out，截取最后的错误信息
            safe_dict["term_out"] = "\n".join(getattr(node, "_term_out", [])[-30:]) if hasattr(node, "_term_out") else ""
            
        return safe_dict

    def _get_hybrid_improve_dict(self, debug_node: Node) -> dict:
        """
        核心优化：当 Debug 成功时，向上追溯寻找最初始的 improve 意图。
        将其与当前 debug 成功的指标与报告融合，伪装成一个成功的 improve 节点进行记忆。
        """
        curr = debug_node.parent
        original_improve_node = None
        
        # 沿着父节点一路往上找，直到找到源头的 improve 节点
        while curr:
            node_type = getattr(curr, "node_type", "")
            # 修复点：加入 draft / init，确保基线建立节点也能被追溯到
            if node_type in ["improve", "draft", "merge"]: 
                original_improve_node = curr
                break
            curr = getattr(curr, "parent", None)
            
        original_plan = getattr(original_improve_node, "plan", "") if original_improve_node else ""
        parent_metric = getattr(original_improve_node, "parent_improve_metric", None) if original_improve_node else None
        
        # 判断这是否是一个从0到1的基线节点
        is_baseline_establishment = (parent_metric is None)
        
        return {
            "id": str(debug_node.id),
            # 将原始策略意图与 Debug 补充的修复信息合并，保留完整的上下文
            "plan": f"[Original Strategy]: {original_plan}\n[Applied Fixes]: {getattr(debug_node, 'plan', '')}",
            "node_type": "improve", # 强行标记为 improve，以便送入 improve_buffer
            "diff_patch":getattr(debug_node, "diff_patch", ""),
            "is_buggy": False,
            "exc_type": "",
            "analysis": getattr(debug_node, "analysis", ""),
            "report": getattr(debug_node, "report", ""),
            "metric": debug_node.metric.value if (hasattr(debug_node, "metric") and debug_node.metric) else None,
            "parent_improve_metric": getattr(original_improve_node, "parent_improve_metric", None) if original_improve_node else None,
            "is_baseline_establishment": is_baseline_establishment, # 新增字段：让 LLM 知道这是从 0 到 1
            "parent_id": original_improve_node.id if original_improve_node else (debug_node.parent.id if getattr(debug_node, "parent", None) else None),
        }

    def add_node(self, node: Node):
        """将新生成的节点加入对应的缓冲池，并根据节点状态拦截无效的幻觉数据"""
        if node.node_type not in ['improve', 'debug']:
            raise ValueError("node_type 必须是 'improve' 或 'debug'")

        # 收集本次需要写入的任务列表
        tasks = []

        if node.node_type == 'improve':
            if getattr(node, 'is_buggy', False):
                # 【防幻觉拦截】：有 Bug 的 Improve 节点直接忽略，绝对不进入 Improve 记忆！
                logger.info(f"[Memory] Improve node {node.id} has bugs. Skipping improve memory insertion to prevent hallucinated negative constraints.")
            else:
                # 只有一次性跑通的 Improve 节点才直接记录
                tasks.append(('improve', self.save_buffer_safely(node)))
                
        elif node.node_type == 'debug':
            # 不论成功还是失败，Debug 节点自身都要进入 Debug Buffer，供 LLM 总结常见的报错模式
            tasks.append(('debug', self.save_buffer_safely(node)))
            
            # 【有效策略打捞】：如果 Debug 成功了，说明曾经报错的那个 Improve 想法终于跑通了！
            if getattr(node, 'is_buggy', False) is False:
                logger.info(f"[Memory] Debug node {node.id} succeeded! Recovering original improve intent into memory.")
                hybrid_dict = self._get_hybrid_improve_dict(node)
                tasks.append(('improve', hybrid_dict))

        # 遍历执行写入与压缩逻辑
        for buffer_type, data_dict in tasks:
            self._process_buffer_append(buffer_type, data_dict)
            

    def _process_buffer_append(self, buffer_type: str, data_dict: dict):
        """负责实际的加锁、文件读写和触发 LLM 压缩逻辑（原 add_node 的后半部分）"""
        file_path = self.improve_json if buffer_type == 'improve' else self.debug_json
        needs_compression = False
        buffer_data_to_compress = []
        
        with self.buffer_lock:
            # 1. 读取现有缓冲池
            with open(file_path, 'r', encoding='utf-8') as f:
                buffer: List[Dict] = json.load(f)
            
            # 2. 加入新节点数据
            buffer.append(data_dict)
            
            # 3. 检查是否达到上限
            if len(buffer) >= self.buffer_limit:
                needs_compression = True
                buffer_data_to_compress = list(buffer)  # 浅拷贝
                
                # 立即清空文件
                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump([], f)
            else:
                # 正常写回积累的数据
                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump(buffer, f, ensure_ascii=False, indent=2)
                    
        if needs_compression:
            # 线程解耦
            threading.Thread(
                target=self._compress_buffer, 
                args=(buffer_type, buffer_data_to_compress),
                daemon=True  # 随主进程退出，防止假死
            ).start()            
    def _compress_buffer(self, node_type: str, buffer_data: List[Dict]):
            """调用LLM对缓冲池进行压缩，并清空缓冲池"""
            print(f"[{node_type.upper()}] 缓冲池达到上限 {self.buffer_limit}，正在触发LLM压缩...")
            
            # 动态选择 Prompt 和 Function Schema
            if node_type == 'improve':
                prompt = self._build_improve_prompt(buffer_data)
                current_func_spec = self.improve_func_spec
            else:
                prompt = self._build_debug_prompt(buffer_data)
                current_func_spec = self.debug_func_spec
            
            last_completion_text = None
            for i in range(3):
                completion_text = query(
                    system_message=prompt,
                    user_message=None,
                    model=self.acfg.feedback.model,
                    temperature=self.acfg.feedback.temp,
                    func_spec=current_func_spec, # 注入对应的 Schema
                    cfg=self.cfg, 
                )
                
                if completion_text:
                    last_completion_text = completion_text
                    break
                    
            if last_completion_text is None:
                logger.info("LLM failed to return a valid response after 3 attempts.")
                last_completion_text = "{}" # 返回空 JSON 避免报错
                
            # 传入 node_type 以便按不同的模板渲染 Markdown
            summary = self.extract_summary(last_completion_text, node_type)
            
            # 持久化保存到 Markdown
            md_file = self.improve_md if node_type == 'improve' else self.debug_md
            with open(md_file, 'a', encoding='utf-8') as f:
                f.write(f"\n## {node_type.upper()} Staged Experience Summary\n")
                f.write(summary + "\n")
                f.write("-" * 40 + "\n")
        
    def extract_summary(self, completion_text: Union[str, dict], node_type: str) -> str:
        """解析 JSON 并根据节点类型渲染不同的 Markdown 模板"""
        try:
            # 兼容 aide backend 自动解析好的 dict 或原始 JSON 字符串
            if isinstance(completion_text, dict):
                data = completion_text
            else:
                data = json.loads(completion_text)
            
            if node_type == 'improve':
                md_output = [
                    f"- ✅ **Positive Guidance**:\n  {data.get('positive_guidance', 'N/A')}",
                    f"- ❌ **Negative Constraints**:\n  {data.get('negative_constraints', 'N/A')}",
                    f"- 🔗 **Synergy Observations**:\n  {data.get('synergy_observations', 'N/A')}",
                    f"- 🎯 **Next Steps**:\n  {data.get('next_steps', 'N/A')}"
                ]
            elif node_type == 'debug':
                md_output = [
                    f"- 🐛 **Frequent Error Patterns**:\n  {data.get('frequent_error_patterns', 'N/A')}",
                    f"- 🛠️ **Standard Fixes**:\n  {data.get('standard_fixes', 'N/A')}",
                    f"- 🛡️ **Defensive Coding Rules**:\n  {data.get('defensive_coding_rules', 'N/A')}"
                ]
            else:
                return "Unknown node_type, cannot render summary."

            return "\n".join(md_output)
            
        except json.JSONDecodeError as e:
            logger.warning(f"JSON Parsing Error: {e}")
            # 强制转为 string 再 strip，防止 completion_text 不是字符串导致崩溃
            if isinstance(completion_text, str):
                return completion_text.strip()
            return str(completion_text)
            
        except Exception as e:
            logger.error(f"Unexpected error formatting summary: {e}")
            # 终极兜底
            return str(completion_text)
    
    def _build_improve_prompt(self, nodes: List[Dict]) -> str:
            nodes_str = json.dumps(nodes, ensure_ascii=False, indent=2)
            return f"""
You are a top-tier Machine Learning Expert and AI Architect. Below are the detailed records of {len(nodes)} nodes generated by an AI Agent during recent "Model/Code Optimization (Improve)" iterations.

Your task is to act as the "Long-term Memory & Search Policy Engine" for the Agent. Analyze these attempts and call the provided function to generate a highly concise, quantitative, and instructive summary.

[Node Data]
{nodes_str}

[Analysis Requirements & Constraints]
1. Quantitative Traceability (CRITICAL): 
   - NEVER use vague qualitative words like "moderate", "slightly better", or "large". 
   - You MUST cite specific metric changes (e.g., "AUC 0.7419 -> 0.7551"), exact parameter values (e.g., "weight_decay=0.01"), and the specific Node IDs to ground your claims.

2. Performance Comparison & Hybrid Evaluation: 
   - Compare the `metric` of each node against its `parent_improve_metric`.
   - Note: Some plans contain `[Original Strategy]` and `[Applied Fixes]`. This means the initial strategy had bugs but was successfully repaired. Evaluate the *combined* algorithmic intent, not the bugs.
   - BASELINE EXCEPTION: If `is_baseline_establishment` is true (or `parent_improve_metric` is null), DO NOT attempt to calculate an improvement delta. Instead, treat this as the "Initial Baseline Establishment" (from 0 to 1). Document the core architecture and strategies that successfully achieved this first working metric.

3. Algorithmic Focus for Pitfalls (Negative Constraints):
   - Only list fundamental ML strategy failures (e.g., incorrect architectures, harmful data augmentations, severe overfitting causing train-val gaps).
   - DO NOT list basic Python syntax errors or library import issues as negative constraints.

4. Method Induction & Synergy:
   - Extract proven combinations. Observe if there are "synergy benefits" (e.g., A alone fails, B alone fails, but A + B together show massive improvement).

5. Strategic Next Steps:
   - Propose 2-4 concrete, incremental ML experiments based strictly on the current successful baselines.
   - Avoid destructive suggestions (like completely changing the data pipeline) if a stable baseline already exists. Suggest hyperparameter tuning, specific augmentations, or architectural tweaks.
"""
    
    def _build_debug_prompt(self, nodes: List[Dict]) -> str:
            nodes_str = json.dumps(nodes, ensure_ascii=False, indent=2)
            return f"""
                    You are a top-tier Senior Software Engineer acting as a strict Code Auditor. 
                    Below are the detailed records of {len(nodes)} historical attempts extracted from an AI Agent's Debug Memory Buffer during a "Code Repair" task.
                    
                    [Memory Buffer Data]
                    {nodes_str}
                    
                    [🚨 CRITICAL AUDIT CONSTRAINTS 🚨]
                    1. ANTI-HALLUCINATION: You MUST base your analysis STRICTLY on the provided Memory Buffer. Do NOT invent, guess, or pull external "best practices" to solve bugs that the Agent failed to fix.
                    2. TRUTH OF is_buggy: 
                       - If `is_buggy = true`, the `plan` and `code` represent a FAILED attempt or symptomatic patch. Do NOT summarize these as valid fixes.
                       - A bug is ONLY considered "Resolved" if there is a specific node in the buffer where `is_buggy = false`.
                    
                    [Analysis Requirements]
                    Please output your analysis in the following structured format:
                    
                    1. Root Cause Tracking: 
                       - Analyze the `term_out` across the buffer to summarize the most frequent errors.
                       
                    2. Verified Fixes (If Any): 
                       - Identify specific nodes where `is_buggy = false`. 
                       - Summarize the EXACT code changes or plans that successfully fixed the bug in those specific nodes. 
                       - If no node has `is_buggy = false`, explicitly state: "No successful fixes found in this memory buffer."
                       
                    3. Failed Attempts & Symptomatic Patches: 
                       - Explicitly identify the bugs that were NEVER resolved in this buffer.
                       - Briefly list what the Agent tried for these bugs that DID NOT work (the symptomatic patches), to prevent repeating the same mistakes in the future.
                       
                    4. Defensive Programming Guidance: 
                       - Extract coding standards to prevent future Agents from writing similar bugs. Base these rules ONLY on the validated fixes (if any) and the identified failures in this buffer.
                    """

    def get_memory_context(self, tag: str) -> str:
            """
            提取指定标签（improve 或 debug）的持久化经验(MD)和当前缓冲池(JSON)内容。
            返回的字符串可以直接放入 prompt 的 dict 中。
            """
            if tag not in ['improve', 'debug']:
                raise ValueError("提取标签必须是 'improve' 或 'debug'")
    
            # 获取对应的文件路径
            md_file = self.improve_md if tag == 'improve' else self.debug_md
            json_file = self.improve_json if tag == 'improve' else self.debug_json
    
            context_lines = []
    
            # ==========================================
            # 1. 提取持久化的历史经验 (Markdown)
            # ==========================================
            context_lines.append(f"### {tag.upper()} 历史经验总结 (Persistent Memory) ###")
            if os.path.exists(md_file):
                with open(md_file, 'r', encoding='utf-8') as f:
                    md_content = f.read().strip()
                    if md_content:
                        context_lines.append(md_content)
                    else:
                        context_lines.append("暂无历史经验。")
            else:
                context_lines.append("暂无历史经验。")
                
            context_lines.append("\n") # 分隔符
    
            # ==========================================
            # 2. 提取近期的动作缓冲池 (JSON)
            # ==========================================
            context_lines.append(f"### {tag.upper()} 近期尝试记录 (Recent Buffer) ###")
            
            # 为了防止在读取时缓冲池正在被写入（虽然目前你的压缩逻辑是安全的，但加锁读取更稳健）
            with self.buffer_lock:
                if os.path.exists(json_file):
                    with open(json_file, 'r', encoding='utf-8') as f:
                        try:
                            buffer_data = json.load(f)
                            if buffer_data:
                                # 转换为易读的 JSON 字符串格式供 LLM 阅读
                                context_lines.append(json.dumps(buffer_data, ensure_ascii=False, indent=2))
                            else:
                                context_lines.append("当前缓冲池为空，暂无未压缩的近期尝试。")
                        except json.JSONDecodeError:
                            context_lines.append("[]")
                else:
                    context_lines.append("当前缓冲池为空。")
    
            # 将所有内容拼接为一个长字符串
            return "\n".join(context_lines)