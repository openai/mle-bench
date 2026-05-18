from .skill import SkillEvolver
import logging
import os
import time
import shutil
import pathlib
from pathlib import Path
from typing import Any, Callable, cast
import requests
from aide.backend import FunctionSpec, query
from aide.utils.response import extract_code, extract_text_up_to_code, wrap_code
from aide.utils.metric import MetricValue, WorstMetricValue
import logging
import os
import time
import shutil
import pathlib
import json  # 建议补充
import threading  # 必须补充，用于 self.lock
from pathlib import Path
from typing import Any, Callable, cast

logger = logging.getLogger("aide")


def get_server_url_list():
    """Return server URL list (env GRADING_SERVER_PORT or default)."""
    server_port = os.getenv("GRADING_SERVER_PORT", "5005")
    return [f"http://127.0.0.1:{server_port}"]


server_url_list = get_server_url_list()


def is_server_online(max_retries=3, timeout=300):
    server_url_list = get_server_url_list()
    retry = 0
    index = 0
    server_url = server_url_list[index]
    while retry < max_retries:
        try:
            response = requests.get(f"{server_url}/health", timeout=timeout)
            if response.status_code == 200:
                logger.info(f"Server {server_url} is online, status code: {response.status_code}")
                return True, server_url
            else:
                logger.warning(f"Server returned non-200 status code: {response.status_code}")
                logger.warning(f"Response body: {response.text[:500]}")
                logger.warning(f"Response headers: {dict(response.headers)}")

        except requests.exceptions.Timeout:
            timeout += 20
            logger.error(f"Connection to {server_url} timed out.")
        except requests.exceptions.ConnectionError:
            logger.error(f"Failed to connect to {server_url}, connection error.")
        except requests.exceptions.RequestException as e:
            logger.error(f"Request failed: {str(e)}")
        except Exception as e:
            logger.error(f"Connection to {server_url} failed.")
        retry += 1
        if retry < max_retries:
            index += 1
            index = index%(len(server_url_list))
            server_url = server_url_list[index]
            logger.info(f"Retrying... ({retry}/{max_retries})")
            time.sleep(1)
    logger.error(f"Server is not online after {max_retries} retries.")
    return False, server_url


def call_validate(exp_id, submission_path, timeout=300, max_retries=3):
    online, server_url = is_server_online()
    retry=0
    while retry < max_retries:
        try:
            if online:
                with open(submission_path, "rb") as f:
                    files = {"file": f}
                    response = requests.post(f"{server_url}/validate", files=files, headers={"exp-id": exp_id}, timeout=timeout)
                response_json = response.json()
                logger.info(f"Server returned : {response.text}")
                if "error" in response_json:
                    return False, response_json.get('details', 'Unknown server error')
                
                # 必须显式检查 is_valid 是否为 True
                if not response_json.get("is_valid", False):
                    return False, response_json.get("result", "Validation failed for the submission")
                
                return True, response_json
            else:
                return False, f"Server at {server_url} is not online"
        except requests.exceptions.Timeout:
            logger.error(f"Connection to {server_url} timed out.")
            timeout += 20
        except requests.exceptions.ConnectionError:
            logger.error(f"Failed to connect to {server_url}, connection error.")
        except requests.exceptions.RequestException as e:
            logger.error(f"Request failed: {str(e)}")
        except Exception as e:
            logger.error(f"Connection to {server_url} failed.")
        retry += 1
        if retry < max_retries:
            logger.info(f"Retrying... ({retry}/{max_retries})")
            time.sleep(1)
        else:
            return False, ""



class evaluator:
    def __init__(self, cfg, task_desc):
        self.skill_evo = SkillEvolver(cfg, task_desc)
        self.cfg = cfg
        self.acfg = cfg.agent
        self.task_desc = task_desc
        self.lock = threading.Lock()
        self.review_func_spec = FunctionSpec(
            name="submit_review",
            description="Analyze code execution results, diagnose bugs, and evaluate machine learning model performance metrics.",
            json_schema={
                "type": "object",
                "properties": {
                    "is_bug": {
                        "type": "boolean", 
                        "description": "Set to True if the code failed to execute, contains syntax errors, or logical bugs."
                    },
                    "summary": {
                        "type": "string",
                        "description": (
                            "CRITICAL INSTRUCTIONS: \n"
                            "1. If is_bug is True: Describe the bug precisely and propose a specific code fix.\n"
                            "2. If is_bug is False: Summarize code quality, GPU efficiency, and model selection. "
                            "Compare train/val metrics to provide evidence of overfitting or generalization. "
                            "Address how to bridge the training-validation gap. \n"
                            "3. Formatting: Verify the 'tool' structure usage. If referring to the Kaggle tool, use the exact name: 'tool'."
                        )
                    },
                    "metric": {
                        "type": ["number", "null"], 
                        "description": "The primary validation metric value. Must be null if is_bug is True."
                    },
                    "lower_is_better": {
                        "type": "boolean", 
                        "description": "Optimization direction: True for metrics like Loss/RMSE, False for Accuracy/F1-score."
                    },
                    "gpu_required": {
                        "type": "boolean", 
                        "description": "Indicates if the task effectively utilized or required GPU acceleration."
                    },
                    "performance_status": {
                        "type": "string",
                        "enum": ["overfitting", "underfitting", "good_generalization", "not_applicable"],
                        "description": "Categorize the model's current training state based on the gap between train and validation metrics."
                    }
                },
                "required": ["is_bug", "summary", "metric", "lower_is_better", "gpu_required", "performance_status"],
            }
        )
        
    def save_submission_snapshot(self, node):
            submission_sample_dir = self.cfg.workspace_dir / "input"
            src_dir = self.cfg.workspace_dir / "submission"
            
            # 你的 csv 路径
            sub_file = src_dir / f"submission_{node.id}.csv"
            # 假设你的代码/solution文件路径，如果后缀或前缀不同请自行调整
            sol_file = src_dir / f"solution_{node.id}.py"
    
            is_validate, output = call_validate(self.cfg.exp_name, sub_file)
            try:
                if not is_validate:
                    logger.error(f"Format Validation Failed for node {node.id}: {output}")
                    node.analysis = f"Important Error: The format of the submission.csv is WRONG:{output}.MUST FIX THIS."
                    node.is_buggy = True
                    node.metric =  WorstMetricValue()
                    return False
                logger.info(f"Format check passed for node {node.id}")
            except Exception as e:
                logger.error(f"Error during format validation for node {node.id}: {str(e)}")
                return False
                
            # 2. 物理保存逻辑
            try:
                dst_dir = self.cfg.log_dir / "best_submission"
                dst_dir.mkdir(parents=True, exist_ok=True) 
                
                # 定义目标文件路径 (Submission)
                sub_01 = dst_dir / "submission01.csv"
                sub_02 = dst_dir / "submission02.csv"
                sub_03 = dst_dir / "submission03.csv"
                
                # 定义目标文件路径 (Solution)
                sol_01 = dst_dir / "best_solution01.py"
                sol_02 = dst_dir / "best_solution02.py"
                sol_03 = dst_dir / "best_solution03.py"
                
                temp_sub_path = dst_dir / f"temp_submission_{node.id}.csv"
                temp_sol_path = dst_dir / f"temp_solution_{node.id}.py"
    
                with self.lock:
                    # --- 1. 滚动更新历史 Top 3 ---
                    # 把原有的 02 退位成 03
                    if sub_02.exists(): os.replace(sub_02, sub_03)
                    if sol_02.exists(): os.replace(sol_02, sol_03)
                    
                    # 把原有的 01 退位成 02
                    if sub_01.exists(): os.replace(sub_01, sub_02)
                    if sol_01.exists(): os.replace(sol_01, sol_02)
    
                    # --- 2. 保存当前最佳为 01 ---
                    # 处理 Submission CSV
                    shutil.copy(sub_file, temp_sub_path)
                    os.replace(temp_sub_path, sub_01)
                    
                    # 处理 Best Solution
                    if sol_file.exists():
                        shutil.copy(sol_file, temp_sol_path)
                        os.replace(temp_sol_path, sol_01)
                    elif hasattr(node, 'code'):
                        # 备用方案：如果你的代码直接保存在 node.code 字符串中
                        with open(sol_01, 'w', encoding='utf-8') as f:
                            f.write(node.code)
                    else:
                        logger.warning(f"无法找到对应的 solution 文件({sol_file}) 或 node.code 属性")
                
                logger.info(f"✅ Successfully saved best submission snapshot (Top 3 tracking) from node {node.id}")
                return True
        
            except Exception as e:
                logger.error(f"Failed to save snapshot from node {node.id}: {str(e)}")
                # 清理可能残留的临时文件
                if 'temp_sub_path' in locals() and temp_sub_path.exists():
                    temp_sub_path.unlink()
                if 'temp_sol_path' in locals() and temp_sol_path.exists():
                    temp_sol_path.unlink()
                return False
            
    def get_eva_content_by_label(self) -> str:
        """
        根据标签拼接路径并读取对应的 .md toolkit 文件
        根目录: aideml/aide/kaggle_prompt
        """
    
        # 定义根目录
        base_path = Path("/inspire/ssd/project/sais-auto-scientist/public/mle_eval")
        # 拼接文件名，例如: aideml/aide/kaggle_prompt/text.md
        file_path = base_path / f"{self.cfg.exp_name}.md"
    
        # 检查文件是否存在并读取
        if file_path.exists():
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                return content
            except Exception as e:
                return f"Error reading file: {str(e)}"
        else:
            return f"File not found: {file_path}"   
        

    def evaluate(self, term_out, node , global_step):
        """
        核心评估函数：综合所有信息，生成审查报告。
        """
        # 1. 获取任务分类与特定规则
        personalized_rules = self.get_eva_content_by_label()

        # 2. 组装通用 Prompt 框架
        prompt = """
        You are an Expert Machine Learning Reviewer auditing code executed by an autonomous AI agent.
        Your goal is to thoroughly analyze the provided terminal output (execution logs) and the source code.
        
        You must evaluate the submission strictly following the DOMAIN-SPECIFIC EVALUATION RULES provided.
        
        CRITICAL DEFINITION OF A BUG (is_bug=true):
        You must flag the execution as a BUG if ANY of the following conditions are met:
        1. Runtime Error: A Python traceback or out-of-memory (OOM) error exists in the logs.
        2. Degenerate Output: Loss becomes NaN, or the model outputs constant/invalid predictions.
        3. Missing Metric: The final evaluation metric is NULL, empty, or nowhere to be found in the terminal output.
        
        If it is a SUCCESS (no errors AND metric is explicitly logged), extract the validation metric and analyze the train/val gap.
        If it is a BUG (including missing metric), you MUST provide a detailed error analysis explaining why the metric calculation failed, was bypassed, or what caused the crash.
        """

        # 修改点：在末尾的指令部分，再次向模型强调“查杀”空 Metric 的行为。
        user_content = f"""
        ### 1. Task Description:
        {self.task_desc}

        ### 2. Domain-Specific Evaluation Rules (Pay close attention):
        {personalized_rules}

        ### 3. Agent Generated Code:
        ```python
        {wrap_code(node.code)}
        ```

        ### 4. Terminal Output (Execution Logs):
        ```text
        {term_out}
        ```
        ### 5. The skill.md you may need:
        {self.skill_evo.get_skill_guidance(self.task_desc, global_step)}
        
        ### 6. FINAL INSTRUCTIONS:
        Analyze the execution comprehensively.
        - Step 1: Scan `term_out` for tracebacks or explicit errors.
        - Step 2: Verify if the expected evaluation metric (e.g., Pearson, F1, Accuracy) was actually calculated and printed in `term_out`.
        - Step 3: If an error is found OR if the metric is completely missing, you MUST mark `is_bug` as true. In your report, explicitly analyze the logical flaw in the code that caused the metric to be missing.
        
        Provide your comprehensive evaluation strictly matching the function schema.
        """

        # 3. 调用 LLM (强制使用你的 FunctionSpec)
        try:
            # 这里的 query/generate 调用方式请替换为你框架内的方法
            response = cast(
                dict,
                query(
                    system_message=prompt,
                    user_message=user_content,
                    func_spec=self.review_func_spec,
                    model=self.acfg.feedback.model,
                    temperature=self.acfg.feedback.temp,
                ),
            )
            return response

        except Exception as e:
            logger.error(f"Evaluation failed to parse: {e}")
            # 降级处理，防止框架崩溃
            return {
                "is_bug": True,
                "summary": f"Evaluation pipeline crashed: {str(e)}",
                "metric": None,
                "lower_is_better": True,
                "gpu_required": False
            }