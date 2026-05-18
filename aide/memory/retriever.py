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
    def __init__(self, all_nodes: List , cfg):
        # ✅ 修复 1：支持冷启动（允许 all_nodes 为空）
        self.nodes = all_nodes if all_nodes else []
        self.node_ids = [str(n.id) for n in self.nodes]
        self.cfg = cfg
        
        # 加载模型 (保持你的原有逻辑)
        model_path = self.cfg.faiss_dir
        if not os.path.exists(model_path):
            print("警告：本地模型未找到，尝试在线加载...")
            self.model = SentenceTransformer('thenlper/gte-small')
        else:
            print(f"正在从本地加载模型: {model_path}")
            self.model = SentenceTransformer(model_path)
            
        # 初始化 BM25 和 FAISS 占位符
        self.bm25 = None
        self.index = None
        
        # 如果有历史节点，则全量构建索引
        if self.nodes:
            self._rebuild_indices()
            print("NodeRetriever 初始化完成，索引已就绪。")
        else:
            print("NodeRetriever 初始化完成，当前记忆库为空 (冷启动)。")

    def _rebuild_indices(self):
        """内部方法：全量重建 BM25 和 FAISS 索引"""
        tokenized_corpus = []
        embeddings_list = []
        
        for n in self.nodes:
            # BM25 处理
            t_text = getattr(n, 'tokenized_text', None)
            if not t_text:
                t_text = self._tokenize(self._build_feature_text(n))
                n.tokenized_text = t_text
            tokenized_corpus.append(t_text)
            
            # FAISS 处理
            emb = getattr(n, 'embedding', None)
            if emb is None:
                emb = self.model.encode(self._build_feature_text(n), convert_to_numpy=True).tolist()
                n.embedding = emb
            embeddings_list.append(emb)

        self.bm25 = BM25Okapi(tokenized_corpus)
        
        embeddings_np = np.array(embeddings_list).astype('float32')
        dimension = embeddings_np.shape[1]
        self.index = faiss.IndexFlatIP(dimension) 
        faiss.normalize_L2(embeddings_np)
        self.index.add(embeddings_np)

    def add_node(self, node):
        """✅ 修复 2：动态向索引中追加新节点"""
        self.nodes.append(node)
        self.node_ids.append(str(node.id))
        
        # 确保新节点有分词和向量
        if not getattr(node, 'tokenized_text', None) or not getattr(node, 'embedding', None):
            text = self._build_feature_text(node)
            node.tokenized_text = self._tokenize(text)
            emb = self.model.encode([text], convert_to_numpy=True).astype('float32')
            node.embedding = emb[0].tolist()

        # BM25 不支持增量更新，直接全量重建（内存和耗时通常可控）
        self._rebuild_indices()

    def _build_feature_text(self, node) -> str:
        clean_text = " ".join(str(getattr(node, 'compressed_text', '')).split())
        clean_framework = " ".join(str(getattr(node, 'core_framework', '')).split())
        return f"compressed_text: {clean_text}\ncompressed_framework: {clean_framework}"

    def _tokenize(self, text: str) -> List[str]:
        return re.findall(r'[a-zA-Z_]\w+|[^\w\s]', text.lower())

    def search_top_k_ids(self, query_node, top_k: int = 3) -> List[str]:
        # ✅ 修复 3：防止在空库时执行检索报错
        if not self.nodes or self.index is None or self.bm25 is None:
            return []
            
        query_text = self._build_feature_text(query_node)
        
        # 1. BM25 路
        query_tokens = self._tokenize(query_text)
        bm25_scores = self.bm25.get_scores(query_tokens)
        bm25_rank = np.argsort(bm25_scores)[::-1]
          
        # 2. FAISS 路
        query_vec = self.model.encode([query_text], convert_to_numpy=True).astype('float32')
        faiss.normalize_L2(query_vec)
        _, faiss_rank = self.index.search(query_vec, min(len(self.nodes), 100))
        faiss_rank = faiss_rank[0]

        # 3. RRF 融合
        rrf_scores = {}
        rrf_constant = 60
        for rank, idx in enumerate(bm25_rank[:100]):
            score = 1.0 / (rrf_constant + rank)
            # 提权：报错类型一致
            if (getattr(self.nodes[idx], 'exec_type', None) == getattr(query_node, 'exec_type', None) 
                and getattr(query_node, 'exec_type', None) is not None):
                score *= 1.2
            rrf_scores[idx] = rrf_scores.get(idx, 0) + score
    
        for rank, idx in enumerate(faiss_rank):
            if idx == -1: continue
            rrf_scores[idx] = rrf_scores.get(idx, 0) + 1.0 / (rrf_constant + rank)
            
        sorted_indices = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
        valid_node_ids = []
        for idx, score in sorted_indices[:top_k]:
            # 转换为整型，防止底层返回 numpy.int64 等类型导致意外
            idx_int = int(idx) 
            
            # 核心防御：检查索引是否合法 (防止 FAISS 返回 -1 或者越界)
            if 0 <= idx_int < len(self.node_ids):
                valid_node_ids.append(self.node_ids[idx_int])
            else:
                import logging
                logger = logging.getLogger("aide")
                logger.warning(f"⚠️ Retriever returned out-of-bounds index: {idx_int}. Total node_ids length: {len(self.node_ids)}. Skipping this candidate.")
                
        return valid_node_ids