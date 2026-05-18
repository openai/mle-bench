import ast
import json
import os
import difflib
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional
from aide.backend import query,FunctionSpec
import re
from .retriever import NodeRetriever
import threading
import logging
from pathlib import Path
logger = logging.getLogger("aide")

@dataclass
class GlobalMemoryEntry:
    id: str
    code: str
    exec_type : str
    compressed_text: str  # LLM 压缩后的自然语言描述
    core_framework: str   # LLM 提取的核心框架代码/伪代码
    tokenized_text: List[str] #分词后的embedding text
    embedding: List[float] = None  # 用于持久化存储语义向量

class GlobalMemoryManager:
    def __init__(self, 
                 cfg,
                 memory_path: str = "global_memory.json", 
                 n_threshold: float = 0.4, 
                 m_threshold: float = 0.85,

                ):
        # 假设 cfg.workspace_dir 是字符串或 Path
        base_dir = Path(cfg.workspace_dir) / "memory"
        
        # 自动创建目录（如果不存在的话），防止后续保存 JSON 时报错
        base_dir.mkdir(parents=True, exist_ok=True)
        
        self.memory_path = base_dir/memory_path
        self.n = n_threshold # 判定不相关的上限
        self.m = m_threshold # 判定一致的下限
        self.memory: List[GlobalMemoryEntry] = self._load_memory()
        self.cfg = cfg
        self.acfg = cfg.agent
        self.retriever = NodeRetriever(self.memory , self.cfg)
        self.memory_lock = threading.Lock()
        self.compress_func_spec = FunctionSpec(
            name="extract_kaggle_code_structure",
            description="对 Kaggle 竞赛代码进行结构化分析与压缩，提取核心特征和逻辑骨架",
            json_schema={
        "type": "object",
        "properties": {
            "summary": {
                "type": "string",
                "description": "概括此代码的核心思路。"
            },
            "features": {
                "type": "array",
                "items": {
                    "type": "string"
                },
                "description": "列表形式，描述特征工程的操作（例如：'Target Encoding', 'Rolling Mean' 等）。"
            },
            "model": {
                "type": "string",
                "description": "描述模型架构及关键超参数。"
            },
            "strategy": {
                "type": "string",
                "description": "描述训练策略（例如：'5-Fold CV', 'Weighted Loss' 等）。"
            },
            "core_framework": {
                "type": "string",
                "description": "提取代码的逻辑骨架，去除具体变量名，保留核心函数调用链。"
            }
        },
        "required": ["summary", "features", "model", "strategy", "core_framework"]
    }
        )
    def _load_memory(self) -> List[GlobalMemoryEntry]:
        if os.path.exists(self.memory_path):
            with open(self.memory_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                return [GlobalMemoryEntry(**item) for item in data]
        return []

    def _save_memory(self):
        with open(self.memory_path, 'w', encoding='utf-8') as f:
            json.dump([asdict(e) for e in self.memory], f, ensure_ascii=False, indent=2)
            
    def extract_summary(self, completion_text) -> dict:
            """
            解析 LLM 通过 Function Calling 返回的结果。
            兼容 aide backend 直接返回 dict 或未解析的 JSON string 的情况。
            """
            try:
                # 【核心修复】：判断类型
                if isinstance(completion_text, dict):
                    extracted_data = completion_text
                else:
                    extracted_data = json.loads(completion_text)
                
                # 进行一次基础的格式校验，确保所有 Key 都存在
                expected_keys = ["summary", "features", "model", "strategy", "core_framework"]
                for key in expected_keys:
                    if key not in extracted_data:
                        extracted_data[key] = "None"
                        
                return extracted_data
                
            except json.JSONDecodeError as e:
                print(f"JSON Parsing Error: Failed to decode LLM output. Error: {e}")
                # 如果解析失败，返回默认空结构，保证系统的健壮性
                return {
                    "summary": "Error: Failed to parse LLM summary.",
                    "features": [],
                    "model": "Unknown",
                    "strategy": "Unknown",
                    "core_framework": "Unknown"
                }
            except Exception as e:
                print(f"Unexpected error during summary extraction: {e}")
                return {
                    "summary": "Error: Unexpected error.",
                    "features": [],
                    "model": "Unknown",
                    "strategy": "Unknown",
                    "core_framework": "Unknown"
                }

    # 1. LLM 结构化压缩
    def compress_code_via_llm(self, code: str) -> Dict[str, str]:
        """
        针对 Kaggle 任务的结构化压缩 Prompt
        """
        prompt = f"""
你是一个资深的 AI 架构师和代码逆向工程专家。你的任务是对传入的 Kaggle 竞赛脚本进行“去业务化”的代码骨架压缩与逆向解析。

你的输出将被用于代码查重和逻辑比对节点。因此，你必须严格保留代码的【函数签名】、【数据流转关系】和【核心控制流】，剔除所有非核心的业务逻辑代码（如具体的 print、基础的数据处理细节）。

【核心要求】
1. 去除所有业务背景描述，将问题抽象为纯数学/机器学习任务（如 Tabular 二分类）。
2. 在 `summary` 字段中，用3-4句话精准概括其数据流派、模型选型和融合策略。
3. 在 `core_framework` 字段中，**必须以“伪代码拓扑”或“调用链图”的形式输出**。这是重中之重，必须让另一台 LLM 能仅根据这段文本，还原出原代码的函数定义和先后执行顺序。

【输出规范指导】
假设输入是一段包含特征生成、交叉验证和后处理融合的代码：

❌ 错误的 core_framework（过于泛泛）：
"加载数据 -> 提取特征 -> 运行 LightGBM 交叉验证 -> 融合预测结果保存。"

✅ 正确的 core_framework（具备可逆推性）：
"
1. GLOBAL DEFINITIONS: 
   - Constants: PATHs, Hyperparameters (LGBM params with early stopping).
2. FUNCTION `create_features(df, is_train)`:
   - Input: DataFrame.
   - Operations: String Split (Categorical extraction), Groupby(mean/sum diff), Row-wise stats(max/min/var), Logical interaction(A * B).
   - Return: Transformed DataFrame.
3. FUNCTION `encode_features(train, test)`:
   - Operations: LabelEncoding on concatenated categorical subset.
   - Return: train, test, encoders_dict.
4. MAIN FLOW `main()`:
   - Load CSV -> Call `create_features` (Train/Test) -> Call `encode_features`.
   - Setup: StratifiedKFold (n_splits=3).
   - Loop over KFold:
       - Train LGBM with validation.
       - Accumulate OOF predictions and Test predictions (test_pred / 3).
   - Post-KFold: Train `final_model` on 100% full data (num_boost_round=2000).
   - Blending logic: `final_predictions = 0.5 * CV_test_preds + 0.5 * Full_model_test_preds`.
   - Save submission.
"

请基于上述严格的逆向工程标准，分析以下代码并返回对应格式的 JSON 结果：
{code}
{code}
"""
        last_completion_text = None
        for i in range(3):
            completion_text = query(
                system_message=prompt,
                user_message=None,
                model=self.acfg.feedback.model,
                func_spec=self.compress_func_spec,
                temperature=self.acfg.feedback.temp,
                cfg=self.cfg, 
            )
            
            if completion_text:
                last_completion_text = completion_text
                break  # 成功拿到结果，跳出循环
        print(f"Attempt {i+1}: LLM returned empty response.")
        
        # 2. 检查循环结束后是否拿到了结果
        if last_completion_text is None:
            logger.info("LLM failed to return a valid response after 3 attempts.")
            last_completion_text = "LLM failed to return a valid response after 3 attempts."
        
        result = self.extract_summary(last_completion_text)
        return result

    # 2. LLM 相似度仲裁
    def normalize_bool(self, x):
        if isinstance(x, bool):
            return x
        if isinstance(x, str):
            return x.lower() == "true"
            
    def llm_judge_similarity(self, code_new: str,code_plan: str, candidates: List[GlobalMemoryEntry]) -> bool:
            if not candidates:
                return False
                
            # 动态拼接所有候选人的代码，打上 ID 标签
            candidates_str = ""
            for i, cand in enumerate(candidates):
                candidates_str += f"\n=== 候选代码 {i+1} (ID: {cand.id}) ===\n{cand.code}\n"
    
            prompt = f"""
    判定以下【新代码】在“核心逻辑与工程实现”上，是否与提供的【历史候选代码列表】中的**任意一个**属于高度重复的冗余迭代。
    
    【判定为 TRUE（即：高度重复，无显著价值）的条件】：
    仅仅包含变量改名、代码块位置移动、无意义的代码风格调整、微小且缺乏逻辑支撑的超参数扰动（如学习率从 0.05 改为 0.051），且没有解决任何实际报错或性能瓶颈。
    
    【判定为 FALSE（即：有效迭代，不重复）的条件】（只要满足以下任意一项即可）：
    1. 算法与特征：引入了新的特征提取逻辑、数据清洗机制，或改变了模型网络结构/融合策略。
    2. 训练机制优化：引入了显著改变训练动态的技术，如梯度累加、混合精度训练 (AMP)、梯度检查点、或新的学习率调度器。
    3. 系统稳定性与修 Bug：修复了致命 Bug（例如修复数据类型转换报错等）。
    4. 监控与工程部署：增加了有效的模型监控指标或保存机制。
    ****如果代码能够完成对应plan中规定的debug操作，无论其他部分有多一致都判定为FALSE****
    
    请仔细比对：
    【新代码】:
    {code_new}
    新代码的执行计划：
    {code_plan}
    
    【历史候选代码列表】:
    {candidates_str}
    
    请判断【新代码】是否与上述候选代码中的某一个高度冗余。
    输出 格式： {{"is_similar": boolean, "duplicate_of_id": "如果是冗余的，请填入对应的历史候选代码ID，否则填None", "reason": "简短判定理由"}}
    """
            last_completion_text = None
    
            for i in range(3):
                completion_text = query(
                    system_message=prompt,
                    user_message=None,
                    model=self.acfg.cheap.model,
                    temperature=self.acfg.cheap.temp,
                    cfg=self.cfg, 
                )
                
                if completion_text:
                    last_completion_text = completion_text
                    break
                print(f"Attempt {i+1}: LLM returned empty response.")
            
            if last_completion_text is None:
                logger.info("LLM failed to return a valid response after 3 attempts.")
                return False
    
            parsed_result = {}
            try:
                if isinstance(last_completion_text, dict):
                    parsed_result = last_completion_text
                elif isinstance(last_completion_text, str):
                    # 【核心修复】：提取大模型回复中的 JSON 实体块
                    # 寻找文本中第一个 '{' 和最后一个 '}' 的位置
                    start_idx = last_completion_text.find('{')
                    end_idx = last_completion_text.rfind('}')
                    
                    # 如果成功找到了括号闭环，则截取这部分内容
                    if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
                        json_str = last_completion_text[start_idx:end_idx+1]
                    else:
                        # 如果没找到，兜底使用原文本（虽然这通常意味着 LLM 完全没有遵循格式）
                        json_str = last_completion_text
                        
                    parsed_result = json.loads(json_str)
                    
            except json.JSONDecodeError as e:
                # 增加对原始文本的日志打印，方便未来 debug 大模型到底返回了什么鬼东西
                logger.warning(f"JSON Parsing Error in batch similarity check. Error: {e}\nRaw LLM Output:\n{last_completion_text}")
                return False
    
            is_sim = parsed_result.get("is_similar", False)

            duplicate_id = parsed_result.get("duplicate_of_id", "Unknown")
            reason = parsed_result.get("reason", "Unknown")
            
            result = self.normalize_bool(is_sim)
            if result:
                logger.info(f"Node is repeating: True, matches old node {duplicate_id} because: {reason}")
            else:
                logger.info(f"Node is unique: False, because: {reason}")
    
            return result
    def process_new_node(self, node):
            # A. 压缩（耗时，完全无锁并行）
            compression = self.compress_code_via_llm(node.code)
            
            query_node = GlobalMemoryEntry(
                id=node.id,
                code=node.code,
                exec_type=node.node_type, 
                compressed_text=compression.get('summary', ''),
                core_framework=compression.get('core_framework', ''),
                embedding=None,       
                tokenized_text=None   
            )
            
            embedding_text = self.retriever._build_feature_text(query_node)
            tokenized_text = self.retriever._tokenize(embedding_text)
            
            emb_np = self.retriever.model.encode([embedding_text], convert_to_numpy=True).astype('float32')
            query_node.embedding = emb_np[0].tolist()
            query_node.tokenized_text = tokenized_text
    
            # B. 获取候选人（极快，加锁保护 FAISS/BM25 读写安全）
            candidates_to_judge = []
            with self.memory_lock:
                if len(self.memory) > 0:
                    top_k_candidates = self.retriever.search_top_k_ids(query_node)
                    for cand_id in top_k_candidates:
                        target_entry = next((e for e in self.memory if e.id == cand_id), None)
                        if target_entry: 
                            candidates_to_judge.append(target_entry)
    
            # C. 深度相似度仲裁（耗时，完全无锁并行）
            is_duplicate = False
            if candidates_to_judge:
                logger.info(f"Refining: Calling Pure LLM to judge {node.id} against {len(candidates_to_judge)} candidates in batch...")
                # 只有这里变了，直接把候选人列表全扔进去
                is_duplicate = self.llm_judge_similarity(node.code, node.plan,candidates_to_judge)
    
            # D. 持久化（极快，加锁写入）
            if not is_duplicate:
                with self.memory_lock:
                    self.memory.append(query_node)
                    self.retriever.add_node(query_node)  # 同步更新检索器的索引
                    self._save_memory()
                
                logger.info(f"New memory saved: {node.id}")
                return True
            else:
                logger.info(f"Node {node.id} rejected as duplicate.")
                return False