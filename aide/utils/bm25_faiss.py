import pyarrow as pa

if not hasattr(pa, 'PyExtensionType'):
    pa.PyExtensionType = pa.ExtensionType

import re
import json
import faiss
import numpy as np
from typing import List
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
import os

class NodeRetriever:
    def __init__(self, cfg,all_nodes: List):
        """初始化时一次性构建索引"""
        self.cfg = cfg
        if not all_nodes:
            raise ValueError("The nodes list for indexing cannot be empty.")
            
        self.nodes = all_nodes
       # ================== 核心修改部分 ==================
        # 替换为你下载模型的实际绝对路径
        model_path = self.cfg.faiss_dir
        
        # 检查路径是否存在，防止路径填错导致再次报错
        if not os.path.exists(model_path):
            print(f"警告：本地模型路径 {model_path} 未找到，将尝试在线加载...")
            self.model = SentenceTransformer('thenlper/gte-small')
        else:
            print(f"正在从本地加载模型: {model_path}")
            self.model = SentenceTransformer(model_path)
        # =================================================
        
        # 1. 构建特征文本库
        self.corpus_texts = [self._build_feature_text(n) for n in all_nodes]
        
        # 2. 准备 BM25 (关键词路)
        tokenized_corpus = [self._tokenize(t) for t in self.corpus_texts]
        self.bm25 = BM25Okapi(tokenized_corpus)
        
        
        # 3. 准备 FAISS (语义路)
        embeddings = self.model.encode(self.corpus_texts, convert_to_numpy=True)
        # --- 核心修复：显式转换为 float32 ---
        embeddings = embeddings.astype('float32') 
        dimension = embeddings.shape[1]
        
        self.index = faiss.IndexFlatIP(dimension)  # 内积索引
        faiss.normalize_L2(embeddings)            # 此时 embeddings 已经是 float32，不会报错
        self.index.add(embeddings)                 # 已经转换过，不需要再 astype 了
        

    def term_out_cut(self, node, max_line=100) -> str:
        """格式化并截断执行输出，保留关键指标和异常信息"""
        lines = node._term_out or []
        lines = [l for l in lines if l.strip() != ""]  # 去掉纯空行

        # --- 1. 关键词保护机制 ---
        critical_keywords = ["Accuracy", "Index", "Gap", "Metric", "Score", "Final"]
        important_lines = []
        for line in reversed(lines):
            if any(key.lower() in line.lower() for key in critical_keywords):
                important_lines.append(line)
            if len(important_lines) >= 10: 
                break
        important_lines.reverse()

        # --- 2. 上下文截断 ---
        tail_lines = lines[-max_line:] if len(lines) > max_line else lines

        # --- 3. 组合去重 ---
        combined_lines = list(dict.fromkeys(important_lines + tail_lines))
        term_out_text = "\n".join(combined_lines)

        # --- 4. 异常信息结构化 ---
        error_header = ""
        if node.exc_type is not None:
            error_header += f"EXCEPTION_TYPE: {node.exc_type}\n"
            if node.exc_info:
                info_str = json.dumps(node.exc_info, default=str)
                error_header += f"EXCEPTION_INFO: {info_str}\n"
            if node.exc_stack:
                stack_str = json.dumps(node.exc_stack, default=str)
                error_header += f"EXCEPTION_STACK: {stack_str}\n"
        
        return error_header + term_out_text

    def _build_feature_text(self, node) -> str:
        """构建用于检索的核心特征文本"""
        term_out = self.term_out_cut(node)
        features = [
            f"CODE: {node.code or ''}",
            f"PLAN: {node.plan or ''}",
            f"TERMINAL_OUTPUT: {term_out}"
            
            
        ]
        return "\n".join(features)

    def _tokenize(self, text: str) -> List[str]:
        """针对代码和日志的正则分词"""
        return re.findall(r'[a-zA-Z_]\w+|[^\w\s]', text)

    def search(self, query_node, top_k: int = 3):
        """执行混合检索并返回 Top K 个 Node 对象"""
        # 必须使用同样的特征构建逻辑
        query_text = self._build_feature_text(query_node)
        
        # --- BM25 路 ---
        bm25_scores = self.bm25.get_scores(self._tokenize(query_text))
        bm25_rank = np.argsort(bm25_scores)[::-1]
          
        # --- FAISS 路 ---
        query_vec = self.model.encode([query_text], convert_to_numpy=True)

        # --- 核心修复：显式转换为 float32 ---
        query_vec = query_vec.astype('float32')

        faiss.normalize_L2(query_vec)
        _, faiss_rank = self.index.search(query_vec, len(self.nodes))
        faiss_rank = faiss_rank[0]

        # --- RRF 融合 ---
        rrf_scores = {}
        rrf_constant = 60
        
        # 融合 BM25 排名 (前100名)
        for rank, idx in enumerate(bm25_rank[:100]):
            score = 1.0 / (rrf_constant + rank)
            # 提权逻辑：报错类型一致时增加 20% 权重
            if (self.nodes[idx].exc_type == query_node.exc_type 
                and query_node.exc_type is not None):
                score *= 1.2
            rrf_scores[idx] = rrf_scores.get(idx, 0) + score
    
        # 融合 FAISS 排名 (前100名)
        for rank, idx in enumerate(faiss_rank[:100]):
            rrf_scores[idx] = rrf_scores.get(idx, 0) + 1.0 / (rrf_constant + rank)
            
        # 最终排序
        sorted_indices = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
        return [self.nodes[idx] for idx, score in sorted_indices[:top_k]]


class RemoteNodeSelector:
    def __init__(self, embedding_dim):
        # 使用简单的暴力 L2 索引以确保距离计算的准确性
        self.index = faiss.IndexFlatL2(embedding_dim)
        self.node_registry = []

    def update_registry(self, good_nodes, embeddings):
        """更新 good_nodes 缓冲池"""
        if len(embeddings) == 0:
            return
        self.index.reset()
        self.index.add(np.array(embeddings).astype('float32'))
        self.node_registry = good_nodes

    def select_most_distant_node(self, best_node_emb, top_n=50):
        """
        在索引中寻找与 best_node 语义差异最大的节点
        """
        best_node_emb = np.array([best_node_emb]).astype('float32')
        
        # 1. 检索出足够多的候选者（或者全量检索）
        search_k = min(len(self.node_registry), top_n)
        if search_k == 0:
            return None

        distances, indices = self.index.search(best_node_emb, search_k)
        
        # 2. 选取距离最大的索引（即 search 结果的最后一个）
        # Faiss 的 search 返回的是升序排列（最近在前），所以取最后一位
        remote_idx = indices[0][-1]
        remote_dist = distances[0][-1]
        
        selected_node = self.node_registry[remote_idx]
        
        return selected_node, remote_dist

# --- 进化触发逻辑控制 ---
def evolutionary_step(best_node, good_nodes, selector, improve_count, stagnation_limit=5):
    """
    基于停滞状态的混合触发逻辑
    """
    # 假设 get_embedding 是您现有的向量化方法
    best_emb = get_embedding(best_node.content)
    good_embs = [get_embedding(n.content) for n in good_nodes]
    
    selector.update_registry(good_nodes, good_embs)
    
    # 策略：如果连续多次没有 Improve，或者以一定基础概率触发
    is_stagnated = (improve_count >= stagnation_limit)
    trigger_prob = 0.3 # 基础探索概率
    
    if is_stagnated or (np.random.rand() < trigger_prob):
        print(f"触发语义远端融合 | 原因: {'停滞' if is_stagnated else '随机探索'}")
        
        target_node, dist = selector.select_most_distant_node(best_emb)
        
        if target_node:
            # 执行交叉进化逻辑
            new_node = crossover_logic(best_node, target_node)
            return new_node
            
    return None