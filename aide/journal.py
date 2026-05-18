"""
The journal is the core datastructure in AIDE that contains:
- the generated code samples
- information how code samples relate to each other (the tree structure)
- code execution results
- evaluation information such as metrics
...
"""
import pyarrow as pa
if not hasattr(pa, 'PyExtensionType'):
    pa.PyExtensionType = pa.ExtensionType
import time
import uuid
from dataclasses import dataclass, field, InitVar
import dataclasses_json  # 加上这一行
from typing import Literal, Optional , List, Union ,Any
from collections import defaultdict
from dataclasses_json import DataClassJsonMixin
from .interpreter import ExecutionResult
from .utils.metric import MetricValue
from .utils.response import trim_long_string
import math
import copy
import logging
import threading
import re
import json
import faiss
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
import os




@dataclass(eq=False)
class Node(DataClassJsonMixin):
    """A single node in the solution tree."""

    # 1. 必填字段（无默认值）放在最顶部
    code: str
    
    # 2. 初始化参数 (InitVar) - 显式加上类型和默认值
    parent: Optional["Node"] = field(default=None, kw_only=True)
    _logical_parents: list["Node"] = field(default=None, kw_only=True)

    # 3. 其他所有带默认值或 field 的字段，统一加上 kw_only=True
    plan: str = field(default=None, kw_only=True)
    report: str = field(default=None, kw_only=True)
    diff_patch: str = field(default="", kw_only=True)
    step: int = field(default=None, kw_only=True)
    id: str = field(default_factory=lambda: uuid.uuid4().hex, kw_only=True)
    ctime: float = field(default_factory=lambda: time.time(), kw_only=True)
    
    # 所有的 field 建议都放在这里
    children: set["Node"] = field(default_factory=set, kw_only=True)

    # ---- execution info ----
    _term_out: list[str] = field(default=None, kw_only=True)  # type: ignore
    exec_time: float = field(default=None, kw_only=True)  # type: ignore
    exc_type: str | None = field(default=None, kw_only=True)
    exc_info: dict | None = field(default=None, kw_only=True)
    exc_stack: list[tuple] | None = field(default=None, kw_only=True)

    bug_mem: str | None = field(default=None, kw_only=True)
    improve_mem: str | None = field(default=None, kw_only=True)

    ablation_block: str | None = field(default=None, kw_only=True)
    gap: float = field(default=None, kw_only=True)

    #用于区分不同的分支,一个根节点id的list
    root: str = field(default=None, kw_only=True)
    

    # ---- evaluation ----
    # post-execution result analysis (findings/feedback)
    analysis: str = field(default=None, kw_only=True)  # type: ignore
    metric: MetricValue = field(default=None, kw_only=True)  # type: ignore
    # whether the agent decided that the code is buggy
    # -> always True if exc_type is not None or no valid metric
    is_buggy: bool = field(default=None, kw_only=True)  # type: ignore
    
    # --- MCTS 核心字段 (直接加进来) ---
    visits: int = field(default=0, kw_only=True)
    total_reward: float = field(default=0.0, kw_only=True)
    uct: float = field(default=0.0, kw_only=True)

    #---- Memory记忆---
    node_type : str = field(default=None, kw_only=True)  # type: ignore
    parent_improve_metric : float = field(default=None, kw_only=True)

    def generate_bug_mem(self) -> str:
        """
        生成 Bug 记忆：
        1. 如果父节点是 Bug，则继承其记忆并追加父节点的失败尝试。
        2. 如果父节点正常（Improvement/Draft），则重置为空。
        """
        # 1. 根节点或没有父节点的情况
        if self.parent is None:
            return ""

        # 2. 如果父节点被标记为 is_buggy，说明当前节点是针对父节点的 Debug 尝试
        if self.parent.is_buggy:
            # 获取父节点之前的 bug_mem (防止 None)
            prev_mem = self.parent.bug_mem if self.parent.bug_mem else ""

            # 将父节点这次失败的计划格式化
            new_failed_entry = (
                f"\n[Previous Failed Debug Attempt]\n"
                f"Failed Plan: {self.parent.plan}\n"
                f"Failed Analysis: {self.parent.analysis}\n"
                f"{'-' * 30}\n"
            )

            # 累加记忆
            return prev_mem + new_failed_entry

        # 3. 如果父节点不为 bug，说明这是从一个成功的基准开启的 Improve 或 Draft 操作
        else:
            # 初始化为空字符串（对应你要求的 bug_mem = []）
            return ""

    def generate_improve_mem(self) -> str:
        """
        生成优化记忆：
        1. 如果父节点是正常的（is_buggy=False），则累积其方案到优化历史中。
        2. 如果父节点是 Bug（is_buggy=True），则当前节点属于 Debug 流程，
           此时应继承父节点的 improve_mem，但不记录该失败的代码。
        """
        if self.parent is None:
            return ""

        # 情况 A: 父节点是成功的方案，但在进行下一步优化
        if not self.parent.is_buggy:
            prev_improve_mem = self.parent.improve_mem if self.parent.improve_mem else ""

            # 记录这次成功的方案，作为后续优化的“基准”和“避免重复项”
            new_entry = (
                f"\n[Past Successful Attempt]\n"
                f"Strategy: {self.parent.plan}\n"
                f"Metric Score: {self.parent.metric}\n"
                f"Code implemented in this version:\n{self.parent.code}\n"
                f"{'=' * 30}\n"
            )
            return prev_improve_mem + new_entry

        # 情况 B: 父节点是 Bug，我们正在 Debug
        else:
            # 继承上一个成功节点的优化记忆，不增加新的内容
            # 确保在 Debug 过程中不丢失之前的优化路线图
            return self.parent.improve_mem if self.parent.improve_mem else ""

    @property
    def stage_name(self) -> Literal["draft", "debug", "improve"]:
        """
        Return the stage of the node:
        - "stage" if the node is an initial solution draft
        - "debug" if the node is the result of a debugging step
        - "improve" if the node is the result of an improvement step
        """
        if self.parent is None:
            return "draft"
        return "debug" if self.parent.is_buggy else "improve"

    def absorb_exec_result(self, exec_result: ExecutionResult):
        """Absorb the result of executing the code from this node."""
        self._term_out = exec_result.term_out
        self.exec_time = exec_result.exec_time
        self.exc_type = exec_result.exc_type
        self.exc_info = exec_result.exc_info
        self.exc_stack = exec_result.exc_stack

    @property
    def term_out(self) -> str:
        """Get the terminal output of the code execution (after truncating it)."""
        return trim_long_string("".join(self._term_out))

    @property
    def is_leaf(self) -> bool:
        """Check if the node is a leaf node in the solution tree."""
        return not self.children

    def __eq__(self, other):
        return isinstance(other, Node) and self.id == other.id

    def __hash__(self):
        return hash(self.id)

    @property
    def debug_depth(self) -> int:
        """
        Length of the current debug path
        - 0 if the node is not a debug node (parent is not buggy)
        - 1 if the parent is buggy but the skip parent isn't
        - n if there were n consecutive debugging steps
        """
        if self.stage_name != "debug":
            return 0
        return self.parent.debug_depth + 1  # type: ignore
        
    def uct_value(self,total_forest_visits, exploration_constant: float = 1.414) -> float:
        
        
        # 确定父节点访问量基数
        actual_parents = self._logical_parents if self._logical_parents else ([self.parent] if self.parent else [])
        if actual_parents:
            parent_visits = max(p.visits for p in actual_parents if p)
        else:
            # 根节点：访问量为总森林访问量
            parent_visits = total_forest_visits
            
        if self.visits == 0:
            return float('inf')

        exploitation = self.total_reward / self.visits
        exploration = exploration_constant * math.sqrt(math.log(parent_visits + 1) / self.visits)
        return exploitation + exploration

    def update_mcts(self, reward: float):
        """反向传播奖励"""
        curr = self
        visited = set()
        queue = [curr]
        while queue:
            node = queue.pop(0)
            if node.id in visited: continue
            visited.add(node.id)
            
            node.visits += 1
            node.total_reward += reward
            
            # 回传路径：优先逻辑父母，次之物理父母
            parents = node._logical_parents if node._logical_parents else ([node.parent] if node.parent else [])
            for p in parents:
                if p: queue.append(p)


@dataclass
class InteractiveSession(DataClassJsonMixin):
    """
    A collection of nodes for an interaction session
    (when the agent interacts with a Jupyter notebook-like interface).
    """

    nodes: list[Node] = field(default_factory=list)
    completed: bool = False

    def append(self, node: Node) -> None:
        node.step = len(self.nodes)
        self.nodes.append(node)

    def generate_nb_trace(self, include_prompt, comment_headers=True) -> str:
        """Generate a trace of the interactive session in IPython format."""
        trace = []
        header_prefix = "## " if comment_headers else ""
        for n in self.nodes:
            trace.append(f"\n{header_prefix}In [{n.step + 1}]:\n")
            trace.append(n.code)
            trace.append(f"\n{header_prefix}Out [{n.step + 1}]:\n")
            trace.append(n.term_out)

        if include_prompt and self.nodes:
            trace.append(f"\n{header_prefix}In [{self.nodes[-1].step + 2}]:\n")

        return "\n".join(trace).strip()


@dataclass
class Journal(DataClassJsonMixin):
    """A collection of nodes representing the solution tree."""

    nodes: list[Node] = field(default_factory=list)
    _model: Any = field(default=None, init=False, repr=False, metadata={'dataclasses_json': {'exclude': lambda x: True}})

    # eda: InteractiveSession = field(default_factory=lambda: InteractiveSession())

    def __post_init__(self):
        """在 dataclass 初始化后加载模型"""
        model_path = '/inspire/ssd/tenant_predefaa-9a1b-4522-bb10-8850f313be13/global_user/5965-xiaokangcheng/models/gte-small'
        
        try:
            from sentence_transformers import SentenceTransformer
            if os.path.exists(model_path):
                print(f"Loading local embedding model from: {model_path}")
                self._model = SentenceTransformer(model_path)
            else:
                print(f"Warning: Local path not found. Loading online gte-small...")
                self._model = SentenceTransformer('thenlper/gte-small')
        except ImportError:
            print("Error: sentence_transformers not installed. Semantic search will be unavailable.")

    def __getitem__(self, idx: int) -> Node:
        return self.nodes[idx]

    def __len__(self) -> int:
        """Return the number of nodes in the journal."""
        return len(self.nodes)

    def append(self, node: Node) -> None:
        """Append a new node to the journal."""
        node.step = len(self.nodes)
        self.nodes.append(node)

    @property
    def draft_nodes(self) -> list[Node]:
        """Return a list of nodes representing intial coding drafts"""
        return [n for n in self.nodes if n.parent is None]

    @property
    def buggy_nodes(self) -> list["Node"]:
        """Return a list of nodes that are considered buggy by the agent,
        excluding deadend nodes and nodes whose ALL children are deadends.
        A bug node with mixed children (some deadend, some not) remains repairable."""
        return [
            n for n in self.nodes
            if n.is_buggy
            and n.node_type != "deadend"
            and not (n.children and all(child.node_type == "deadend" for child in n.children))
        ]

    @property
    def good_nodes(self) -> list["Node"]:
        """Return a list of nodes that are not considered buggy by the agent, 
        excluding deadend nodes and any node that has a deadend child."""
        return [
            n for n in self.nodes 
            if not n.is_buggy 
            and n.node_type != "deadend" 
            and not any(child.node_type == "deadend" for child in n.children)
        ]

    def get_metric_history(self) -> list[MetricValue]:
        """Return a list of all metric values in the journal."""
        return [n.metric for n in self.nodes]

    def get_best_node(self, only_good=True):
        nodes = self.good_nodes if only_good else self.nodes
        nodes = [n for n in nodes if isinstance(n.metric, MetricValue)]
        if not nodes:
            return None
        return max(nodes, key=lambda n: n.uct)

    def generate_summary(self, include_code: bool = False) -> str:
        """Generate a summary of the journal for the agent."""
        summary = []
        for n in self.good_nodes:
            summary_part = f"Design: {n.plan}\n"
            if include_code:
                summary_part += f"Code: {n.code}\n"
            summary_part += f"Results: {n.analysis}\n"
            summary_part += f"Validation Metric: {n.metric.value}\n"
            summary.append(summary_part)
        return "\n-------------------------------\n".join(summary)

    def generate_all_summary(self, include_code: bool = False) -> str:
        """Generate a summary of the journal for the agent."""
        summary = []
        for n in self.nodes:
            summary_part = f"Design: {n.plan}\n"
            if include_code:
                summary_part += f"Code: {n.code}\n"
            summary_part += f"Results: {n.analysis}\n"
            summary_part += f"Validation Metric: {n.metric.value}\n"
            summary.append(summary_part)
        return "\n-------------------------------\n".join(summary)

    def generate_bug_memory(self) -> str:
        # 综合所有错误节点的plan和报错结果
        summary = []
        for n in self.buggy_nodes:
            summary_part = f"Design: {n.plan}\n"
            summary_part += f"Results: {n.analysis}\n"
            summary.append(summary_part)
        return "\n-------------------------------\n".join(summary)
    
    def generate_check_memory(self) -> str:
        # 综合所有错误节点的plan和报错结果
        summary = []
        for n in self.nodes:
            summary_part = f"Check Reports: {n.report}\n"
            summary.append(summary_part)
        return "\n-------------------------------\n".join(summary)


    def different_root_nodes(self) -> List[List[Node]]:
        """
        根据节点的 root 属性，将森林中的节点分组为不同的树。

        Returns:
            List[List[Node]]: 返回一个列表，其中每个元素也是一个列表，包含属于同一棵树的所有节点。
        """
        # 使用 defaultdict 方便分组，key 为 root 的 tuple (因为 list 不可哈希)，value 为节点列表
        trees_dict = defaultdict(list)
        # 假设 self.nodes 存储了所有的节点 (如果是其他变量名，请相应修改，如 self.population)
        # 如果 self 本身就是节点列表的管理者，可以直接迭代 self.nodes
        all_nodes = getattr(self, "nodes", [])

        for node in all_nodes:
            # 健壮性检查：跳过没有 root 属性或 root 为空的节点
            if not node.root:
                continue

            # 将 list 转换为 tuple 以作为字典的 key
            # 这样即使 root 是 ['root_id_1'] 和 ['root_id_2'] 也能被正确区分
            root_key = tuple(node.root)

            trees_dict[root_key].append(node)

        # 将字典的 values 转换为列表返回，不需要 key
        return list(trees_dict.values())

    def different_best_root_nodes(self, list_tree: List[List[Node]]) -> List[Node]:
        """获取每棵树的最佳节点，跳过不存在有效（非 Worst）节点的树"""
        best_nodes = []

        for group in list_tree:
            # 过滤掉坏节点：只保留 metric 不是 WorstMetricValue 且 value 不为 None 的节点
            valid_nodes = [n for n in group if n.metric and not n.metric.is_worst]

            if not valid_nodes:
                # 如果这棵树一个正确的节点都没有，直接跳过
                continue

            # 在有效节点中找出最好的那个
            best_group_node = max(valid_nodes, key=lambda n: n.uct)
            best_nodes.append(best_group_node)

        return best_nodes

    def get_topk_best_nodes(self, k: int, list_node: List[Node]) -> List[Node]:
        """获取前 k 个最佳节点。如果不足 k 个，则返回全部有效节点。"""

        # 过滤出有效节点
        nodes = [n for n in list_node if isinstance(n.metric, MetricValue) and not n.metric.is_worst]

        if not nodes:
            return []  # 返回空列表比返回 None 更健壮，方便后续 for 循环

        # 排序：最好（Better）的在前
        sorted_nodes = sorted(nodes, key=lambda n: n.uct, reverse=True)

        # Python 的切片 sorted_nodes[:k] 具有鲁棒性：
        # 如果 len(sorted_nodes) < k，它会自动返回 [0:len(nodes)]，即全部节点
        return sorted_nodes[:k]

    def diff_most_node(self, target_node: Node, top_k: int = 1) -> Node:
        """
        在全局节点中，寻找与 target_node 语义差距（L2 距离）最大的节点。
        """
        if self._model is None:
            raise RuntimeError("Embedding model is not initialized.")

        # 1. 准备候选节点（排除目标节点本身，且只考虑有意义的节点）
        candidate_nodes = [n for n in self.good_nodes if n.id != target_node.id]
        
        if not candidate_nodes:
            return []

        # 2. 提取文本特征 (推荐优先使用 plan，因为它代表算法逻辑)
        def get_text(n): return n.plan if n.plan else (n.code[:500] if n.code else "")
        
        target_text = get_text(target_node)
        candidate_texts = [get_text(n) for n in candidate_nodes]

        # 3. 生成 Embedding 并转换为 float32
        target_emb = self._model.encode([target_text], convert_to_numpy=True).astype('float32')
        candidate_embs = self._model.encode(candidate_texts, convert_to_numpy=True).astype('float32')

        # 4. 构建 FAISS 索引 (L2 距离)
        dimension = candidate_embs.shape[1]
        index = faiss.IndexFlatL2(dimension)
        index.add(candidate_embs)

        # 5. 搜索全部距离并取最远者
        num_cands = len(candidate_nodes)
        distances, indices = index.search(target_emb, num_cands)

        # 索引 0 指向 target_emb 的搜索结果
        # indices[0] 是按距离从小到大排序的，取最后 top_k 个即为最远
        farthest_indices = indices[0][-top_k:][::-1]
        
        return candidate_nodes[farthest_indices[0]]