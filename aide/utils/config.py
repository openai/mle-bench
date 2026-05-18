"""configuration and setup utils"""

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Hashable, cast, Literal

import rich
from omegaconf import OmegaConf
from rich.syntax import Syntax
import shutup
from rich.logging import RichHandler
import logging
from openai import OpenAI
from aide.journal import Journal

from . import copytree, preproc_data, serialize

shutup.mute_warnings()
logger = logging.getLogger("aide")

""" these dataclasses are just for type hinting, the actual config is in config.yaml """


@dataclass
class StageConfig:
    model: str
    temp: float
    base_url: str
    api_key: str


@dataclass
class LinearDecay:
    alpha: float


@dataclass
class ExponentialDecay:
    gamma: float


@dataclass
class PiecewiseDecay:
    alpha: float
    phase_ratios: list


@dataclass
class DynamicPiecewiseDecay:
    alpha: float
    phase_ratios: list


@dataclass
class ForcedExplodeConfig:
    enabled: bool
    patience: int
    temperature: float
    amplitude: float
    num_sparks: int
    cooldown: int
    stagnation_threshold_pct: float  # percentage threshold for stagnation detection (e.g., 3.0 means 3%)


@dataclass
class DecayConfig:
    decay_type: str
    exploration_constant: float
    lower_bound: float
    linear_decay: LinearDecay
    exponential_decay: ExponentialDecay
    piecewise_decay: PiecewiseDecay
    dynamic_piecewise_decay: DynamicPiecewiseDecay


@dataclass
class SearchConfig:
    max_debug_depth: int
    ablation_improve_depth: int
    debug_prob: float
    topk_best: int
    explosion_prob: float
    crossover_prob: float
    min_sparks: int
    max_sparks: int
    total_sparks: int
    num_drafts: int
    invalid_metric_upper_bound: int
    metric_improvement_threshold: float
    back_debug_depth: int
    num_bugs: int
    num_improves: int
    max_improve_failure: int
    parallel_search_num: int
    forced_explode: ForcedExplodeConfig


@dataclass
class AgentConfig:
    steps: int
    time_limit: int
    k_fold_validation: int
    expose_prediction: bool
    data_preview: bool
    obfuscate: bool
    check_format: bool
    save_all_submission: bool
    steerable_reasoning: bool
    ablation_step: int
    draft: StageConfig
    debug: StageConfig
    feedback: StageConfig
    cheap: StageConfig
    search: SearchConfig
    decay: DecayConfig


@dataclass
class ExecConfig:
    timeout: int
    agent_file_name: str
    format_tb_ipython: bool


@dataclass
class ReportConfig:
    model: str
    temp: float


@dataclass
class Config(Hashable):
    data_dir: Path

    desc_file: Path | None
    generate_report: bool

    goal: str | None
    eval: str | None

    log_dir: Path
    log_level: str
    workspace_dir: Path

    preprocess_data: bool
    copy_data: bool

    exp_name: str

    ablation_prob: float

    report: ReportConfig

    exec: ExecConfig
    agent: AgentConfig
    start_cpu_id: int
    cpu_number: int
    faiss_dir: str


def _get_next_logindex(dir: Path) -> int:
    """Get the next available index for a log directory."""
    max_index = -1
    for p in dir.iterdir():
        try:
            if current_index := int(p.name.split("-")[0]) > max_index:
                max_index = current_index
        except ValueError:
            pass
    return max_index + 1


def _load_cfg(
        path: Path = Path(__file__).parent / "config.yaml", use_cli_args=True
) -> Config:
    cfg = OmegaConf.load(path)
    if use_cli_args:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_cli())
    return cfg


def load_cfg(path: Path = Path(__file__).parent / "config.yaml") -> Config:
    """Load config from .yaml file and CLI args, and set up logging directory."""
    return prep_cfg(_load_cfg(path))


def prep_cfg(cfg: Config):
    if cfg.data_dir is None:
        raise ValueError("`data_dir` must be provided.")

    if cfg.desc_file is None and cfg.goal is None:
        raise ValueError(
            "You must provide either a description of the task goal (`goal=...`) or a path to a plaintext file containing the description (`desc_file=...`)."
        )

    if cfg.data_dir.startswith("example_tasks/"):
        cfg.data_dir = Path(__file__).parent.parent / cfg.data_dir
    cfg.data_dir = Path(cfg.data_dir).resolve()

    if cfg.desc_file is not None:
        cfg.desc_file = Path(cfg.desc_file).resolve()

    top_log_dir = Path(cfg.log_dir).resolve()
    top_log_dir.mkdir(parents=True, exist_ok=True)

    top_workspace_dir = Path(cfg.workspace_dir).resolve()
    top_workspace_dir.mkdir(parents=True, exist_ok=True)

    # generate experiment name using numeric index
    if cfg.exp_name is None:
        # Find existing numeric experiment directories
        existing_numbers = []
        if top_log_dir.exists():
            for item in top_log_dir.iterdir():
                if item.is_dir():
                    try:
                        # Try to parse directory name as integer
                        num = int(item.name)
                        existing_numbers.append(num)
                    except ValueError:
                        # Not a numeric name, skip it
                        pass
        # Generate next number (start from 1 if no existing numbers)
        next_number = max(existing_numbers) + 1 if existing_numbers else 1
        cfg.exp_name = str(next_number)

    cfg.log_dir = (top_log_dir / cfg.exp_name).resolve()
    cfg.workspace_dir = (top_workspace_dir / cfg.exp_name).resolve()

    # validate the config
    cfg_schema: Config = OmegaConf.structured(Config)
    cfg = OmegaConf.merge(cfg_schema, cfg)

    return cast(Config, cfg)


def print_cfg(cfg: Config) -> None:
    rich.print(Syntax(OmegaConf.to_yaml(cfg), "yaml", theme="paraiso-dark"))


def load_task_desc(cfg: Config):
    """Load task description from markdown file or config str."""

    # either load the task description from a file
    if cfg.desc_file is not None:
        if not (cfg.goal is None and cfg.eval is None):
            logger.warning(
                "Ignoring goal and eval args because task description file is provided."
            )

        with open(cfg.desc_file) as f:
            return f.read()

    # or generate it from the goal and eval args
    if cfg.goal is None:
        raise ValueError(
            "`goal` (and optionally `eval`) must be provided if a task description file is not provided."
        )

    task_desc = {"Task goal": cfg.goal}
    if cfg.eval is not None:
        task_desc["Task evaluation"] = cfg.eval

    return task_desc


def prep_agent_workspace(cfg: Config):
    """Setup the agent's workspace and preprocess data if necessary."""
    (cfg.workspace_dir / "input").mkdir(parents=True, exist_ok=True)
    (cfg.workspace_dir / "working").mkdir(parents=True, exist_ok=True)
    (cfg.workspace_dir / "submission").mkdir(parents=True, exist_ok=True)
    (cfg.workspace_dir / "memory").mkdir(parents=True, exist_ok=True)
    copytree(cfg.data_dir, cfg.workspace_dir / "input", use_symlinks=not cfg.copy_data)
    
    if cfg.preprocess_data:
        preproc_data(cfg.workspace_dir / "input")


def save_run(cfg: Config, journal: Journal):
    cfg.log_dir.mkdir(parents=True, exist_ok=True)

    # save journal
    serialize.dump_json(journal, cfg.log_dir / "journal.json")
    # save config
    OmegaConf.save(config=cfg, f=cfg.log_dir / "config.yaml")


def concat_logs(chrono_log: Path, best_node: Path, journal: Path):
    content = (
        "The following is a concatenation of the log files produced.\n"
        "If a file is missing, it will be indicated.\n\n"
    )

    content += "---First, a chronological, high level log of the ml-master run---\n"
    content += output_file_or_placeholder(chrono_log) + "\n\n"

    content += "---Next, the ID of the best node from the run---\n"
    content += output_file_or_placeholder(best_node) + "\n\n"

    content += "---Finally, the full journal of the run---\n"
    content += output_file_or_placeholder(journal) + "\n\n"

    return content


def output_file_or_placeholder(file: Path):
    if file.exists():
        if file.suffix != ".json":
            return file.read_text()
        else:
            return json.dumps(json.loads(file.read_text()), indent=4)
    else:
        return f"File not found."