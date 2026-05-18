import os
import logging
import random
import pathlib
from pathlib import Path
from openai import OpenAI
from aide.backend import query
logger = logging.getLogger("aide")

class SkillEvolver:
    def __init__(self, cfg, task_desc):
        self.cfg = cfg
        self.acfg = cfg.agent
        self.task_desc = task_desc
        self.log_dir = Path(cfg.log_dir)
        self.skill_path = Path(f"/inspire/ssd/tenant_predefaa-9a1b-4522-bb10-8850f313be13/global_user/5965-xiaokangcheng/aideml/aide/kaggle_prompt/evo_skill/{cfg.exp_name}.md")
        self.skill_reverse_path = Path(f"/inspire/ssd/tenant_predefaa-9a1b-4522-bb10-8850f313be13/global_user/5965-xiaokangcheng/aideml/aide/kaggle_prompt/evo_skill/{cfg.exp_name}_reverse.md")
        self.best_code_path = self.log_dir / "best_solution.py"
        self.last_summarized_phase = 0
        self.kaggle_label = self.classify_kaggle_task(self.task_desc)
        self.kaggle_md = self.get_toolkit_content_by_label(self.kaggle_label)
        self.kaggle_tool = self.kaggle_md
    def classify_kaggle_task(self,task_desc) -> str | None:
        """
        使用 Qwen API 根据任务描述进行分类
        """

        # 构造引导 Prompt，确保 LLM 严格遵守输出格式
        system_instruction = (
            "You are a helpful assistant that classifies machine learning tasks. "
            "Analyze the user's competition description and output exactly one label from this list: "
            "['nlp', 'audio', 'img','science','tabular','medical','segmentation','mixed','else']. If the description does not fit any of these, output 'None'. "
            "Below is some classification logic and some example:"
            "1 :nlp: Tasks relying primarily on DeBERTa-v3, RoBERTa or LLMs for semantic understanding. Semantics & QA: google-quest-challenge, tensorflow2-question-answering, chaii-hindi-and-tamil-question-answering, tweet-sentiment-extraction, us-patent-phrase-to-phrase-matching. Content Evaluation & Text Mining: learning-agency-lab-automated-essay-scoring-2, lmsys-chatbot-arena, spooky-author-identification, random-acts-of-pizza. Safety & Standardization: jigsaw-toxic-comment, jigsaw-unintended-bias, detecting-insults, text-normalization-challenge-english/russian. Code Logic (special handling): AI4Code."
            "2 :img: General CV classification based on ImageNet pre-trained models. Basic classification: dogs-vs-cats, dog-breed-identification, leaf-classification, cassava-leaf-disease. Fine-grained recognition (FGVC): herbarium-2020/21/22, inaturalist-2019, iwildcam-2019/20, plant-pathology-2020/21, aerial-cactus-identification. Special vision: alaska2-image-steganalysis (steganography), statoil-iceberg-classifier (radar images)."
            "3 :medical: Medical Imaging - Requires medical pre-trained weights or MIL (Multiple Instance Learning) strategies. aptos2019-blindness-detection, histopathologic-cancer-detection, rsna-breast-cancer-detection, rsna-miccai-brain-tumor, siim-isic-melanoma, ranzcr-clip-catheter, osic-pulmonary-fibrosis-progression."
            "4 :segmentation: Object Detection & Segmentation - Tasks involving coordinate regression or pixel-level classification. Object detection: 3d-object-detection-for-autonomous-vehicles, kuzushiji-recognition, siim-covid19-detection, vinbigdata-chest-xray-abnormalities. Image segmentation: google-research-identify-contrails, hubmap-kidney-segmentation, tgs-salt-identification, uw-madison-gi-tract-segmentation, vesuvius-challenge-ink-detection."
            "5 :tabular: Tabular data - Pure feature engineering driven, best suited for LightGBM / XGBoost / CatBoost. tabular-playground-series-dec-2021, tabular-playground-series-may-2022, new-york-city-taxi-fare-prediction."
            "6 :science: Scientific Discovery & Signal Processing - Involves physical or biological prior knowledge. High Energy Physics / Earth Science: icecube-neutrinos-in-deep-ice, predict-volcanic-eruptions-ingv-oe. Molecules & Materials: champs-scalar-coupling, nomad2018-predict-transparent-conductors, bms-molecular-translation. Biological Sequences & Signals: stanford-covid-vaccine (mRNA), hms-harmful-brain-activity (EEG signals), smartphone-decimeter-2022 (GNSS signals)."
            "7 :audio: Audio Processing - Usually involves conversion to Mel-spectrogram followed by image processing. freesound-audio-tagging-2019, mlsp-2013-birds, tensorflow-speech-recognition, the-icml-2013-whale-challenge."
            "8 :mixed: Multi-modal & Complex Dynamics - h-and-m-personalized-fashion-recommendations (recommendation algo), nfl-player-contact-detection (video+sensors), multi-modal-gesture-recognition, ventilator-pressure-prediction (time sequence control), petfinder-pawpularity-score (image+tabular metadata)."
            "9 :else: If you think the task doesn't belong to any type above, output 'else'."
            "Do not include any explanation or other words."
        )

        try:
            summary = query(
        system_message=system_instruction,  # 你的系统提示词
        user_message=f"Task Description: {task_desc}",    # 你的用户提示词 (不要传 None)
        temperature=0,
        model=self.acfg.cheap.model,
        cfg=self.cfg
    )
            # 清理响应中的空格和换行
            label = summary.strip().lower()

            valid_labels = ["nlp", "audio", "img","science","tabular","medical","segmentation","mixed","else"]
            logger.info(f"label is {label}")
            if label in valid_labels:
                return label
            else:
                return "else"

        except Exception as e:
            print(f"API Error: {e}")
            return None


    def get_toolkit_content_by_label(self,label: str | None) -> str:
        """
        根据标签拼接路径并读取对应的 .md toolkit 文件
        根目录: aideml/aide/kaggle_prompt
        """
        if label is None:
            return "No matching toolkit found for this task type."
    
        # 定义根目录
        base_path = Path("/inspire/ssd/tenant_predefaa-9a1b-4522-bb10-8850f313be13/global_user/5965-xiaokangcheng/aideml/aide/kaggle_prompt")
        # 拼接文件名，例如: aideml/aide/kaggle_prompt/text.md
        file_path = base_path / f"{label}.md"
    
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

    def get_skill_guidance(self, task_desc, global_step):
            """
            核心控制逻辑：根据 global_step 动态调整概率
            """
            # 计算进度系数 (0 到 1)
            total_steps = self.cfg.agent.steps
            step_interval = max(1, total_steps // 4)
            current_phase = global_step // step_interval
    
            # 水位线触发逻辑：确保每个阶段只总结一次
            if current_phase > self.last_summarized_phase and current_phase > 0:
                logger.info(f"Evolution checkpoint at step {global_step}, updating skill.md")
                summary = self.summarize_best_to_md()
                reverse_summary = self.generate_reverse_skill(summary)
                self.last_summarized_phase = current_phase
                
            progress = global_step / total_steps
            # 动态概率模型：前期鼓励探索，后期鼓励收敛（利用 Skill）
            diversity_prob = 0.2 * progress
            exploit_prob = 0.7 + (0.2 * progress) 
            free_prob = 0.1 
            
            r = random.random()
            
            # 1. 安全地获取内容，设置默认值为 kaggle_tool
            current_skill = self.kaggle_tool
            reverse_skill = self.kaggle_tool
    
            if self.skill_path.exists():
                with open(self.skill_path, 'r', encoding='utf-8') as f:
                    current_skill = f.read()
                    
            if self.skill_reverse_path.exists():
                with open(self.skill_reverse_path, 'r', encoding='utf-8') as f:
                    reverse_skill = f.read()
    
            # 2. 根据随机数决定输出哪种 Guidance
            if r < diversity_prob:
                logger.info("reverse thinking mode")
                if self.best_code_path.exists() and self.skill_reverse_path.exists():
                    return reverse_skill
                else:
                    return self.kaggle_tool
            
            elif r < exploit_prob:
                # 利用模式：如果存在 best_code，则返回当前提取的 skill；否则返回基础 kaggle_tool
                logger.info("best skill mode")
                if self.best_code_path.exists() and self.skill_path.exists():
                    return current_skill
                else:
                    return self.kaggle_tool
                    
            else:
                # 自由探索模式
                logger.info("free explore mode")
                return "EXPLORATION_MODE: Ignore previous conventions. Try a completely new approach or unconventional feature engineering."

    def summarize_best_to_md(self):
        """
        将 best_solution.py 抽象为结构化的 skill.md
        """
        if not self.best_code_path.exists():
            return "No best solution found yet."

        with open(self.best_code_path, 'r') as f:
            code_content = f.read()
            
        system_instruction = (
            "You are a Senior Machine Learning Scientist and Kaggle Grandmaster. "
            "Your task is to analyze the current State-of-the-Art (SOTA) Python solution "
            "for a specific ML task and extract reusable 'Skills' and 'Insights'.\n\n"
            "CRITICAL RULES:\n"
            "1. DO NOT just list what the code does (e.g., 'Used LightGBM').\n"
            "2. Focus on the ML RATIONALE: WHY did this specific approach work for THIS dataset? "
            "What hidden data characteristics did it exploit?\n"
            "3. Extract the GENERAL PRINCIPLE so future agents can apply variants of it, rather than copying it blindly.\n\n"
            "For each aspect (Data Processing, Feature Engineering, Model, Validation), structure your Markdown as:\n"
            "- **Observation/Action:** [Briefly what was done]\n"
            "- **ML Rationale (Why it works):** [The underlying mathematical or data-driven reason]\n"
            "- **Guiding Principle:** [How future models should approach this aspect]"
        )
        
        user_content = (
            f"--- Task Description ---\n{self.task_desc}\n\n"
            f"--- Current Best Solution Code ---\n{code_content}\n\n"
            "Please analyze the code above in the context of the task and provide the structured Skill Summary."
        )
        
        try:
            summary = query(
        system_message=system_instruction,  # 你的系统提示词
        user_message=user_content,    # 你的用户提示词 (不要传 None)
        temperature=0.5,
        model=self.acfg.feedback.model,
        cfg=self.cfg
    )

            # 因为没有传 func_spec，所以 response_text 直接就是总结好的字符串
            if summary:
                logger.info(f"Summarize the memory output is: {summary}")
            else:
                logger.error("LLM returned empty response or failed.")
            
            # 写入文件
            with open(self.skill_path, 'w', encoding='utf-8') as f:
                f.write(summary)
            return summary
        except Exception as e:
            return f"Error summarizing skill: {e}"

    def generate_reverse_skill(self, current_skill):
        """
        生成与当前 Skill 路径完全相反的尝试建议（用于打破局部最优）
        """
        system_instruction = (
            "You are a 'Devil's Advocate' for Data Science. Given a current winning strategy, "
            "propose a completely DIFFERENT or OPPOSITE direction. For example, if the skill suggests "
            "Gradient Boosting, suggest Deep Learning. If it suggests filling NAs with 0, suggest "
            "Iterative Imputation. Provide brief, actionable counter-advice."
        )
        try:
            summary = query(
        system_message=system_instruction,  # 你的系统提示词
        user_message=f"Current Strategy:\n{current_skill}",
        temperature=1,
        model=self.acfg.feedback.model,
        cfg=self.cfg
    )

            # 因为没有传 func_spec，所以 response_text 直接就是总结好的字符串
            if summary:
                logger.info(f"Summarize the memory reverse output is: {summary}")
            else:
                logger.error("LLM returned empty response or failed.")
            with open(self.skill_reverse_path, 'w', encoding='utf-8') as f:
                f.write(summary)
            return summary
        except Exception as e:
            return f"Error reverse summarizing skill: {e}"