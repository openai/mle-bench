import logging
from pathlib import Path

logger = logging.getLogger("aide")

class SkillEvolver:
    def __init__(self, cfg, task_desc):
        self.cfg = cfg
        self.task_desc = task_desc
        path_ipy = Path(f"/inspire/ssd/project/sais-auto-scientist/public/mle-skills/{cfg.exp_name}-blueprint.md")
        path_txt = Path(f"/inspire/ssd/project/sais-auto-scientist/public/mle-skills/{cfg.exp_name}.md")
        # 按照 cfg.exp_name 匹配对应的 .md 文件
        if path_txt.exists:
            self.skill_path = path_txt
        else:
            self.skill_path = path_ipy
        
        # 初始化时直接加载固定的 SOTA skill 内容
        self.sota_skill = self._load_sota_skill()

    def _load_sota_skill(self) -> str:
        """
        读取指定的 .md 文件作为固定的 SOTA Skill。
        """
        if self.skill_path.exists():
            try:
                with open(self.skill_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                logger.info(f"Successfully loaded SOTA skill from {self.skill_path}")
                return content
            except Exception as e:
                error_msg = f"Error reading SOTA skill file: {e}"
                logger.error(error_msg)
                return error_msg
        else:
            warning_msg = f"SOTA skill file not found: {self.skill_path}"
            logger.warning(warning_msg)
            return "No SOTA skill found for this experiment. Please check the exp_name and file path."

    def get_skill_guidance(self, task_desc=None, global_step=None):
        """
        固定返回匹配到的 SOTA skill。
        保留 task_desc 和 global_step 参数以兼容外部原有的调用方式，但内部不再做动态概率计算。
        """
        logger.info("Using fixed SOTA skill mode.")
        return self.sota_skill