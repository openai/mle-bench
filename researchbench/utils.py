import difflib
import importlib
import json
import logging
import os
import sys
import time
import uuid
import zipfile
from logging import Logger
from pathlib import Path
from typing import Any, Callable, Optional

import pandas as pd
import py7zr
import yaml
from pandas import DataFrame
from tqdm.auto import tqdm


def purple(str: str) -> str:
    return f"\033[1;35m{str}\033[0m"


def authenticate_kaggle_api() -> "KaggleApi":
    """Authenticates the Kaggle API and returns an authenticated API object, or raises an error if authentication fails."""
    try:
        # only import when necessary; otherwise kaggle asks for API key on import
        from kaggle.api.kaggle_api_extended import KaggleApi

        api = KaggleApi()
        api.authenticate()
        api.competitions_list()  # a cheap op that requires authentication
        return api
    except Exception as e:
        logger.error(f"Authentication failed: {str(e)}")
        raise PermissionError(
            "Kaggle authentication failed! Please ensure you have valid Kaggle API credentials "
            "configured. Refer to the Kaggle API documentation for guidance on setting up "
            "your API token."
        ) from e


def read_jsonl(file_path: str, skip_commented_out_lines: bool = False) -> list[dict]:
    """
    Read a JSONL file and return a list of dictionaries of its content.

    Args:
        file_path (str): Path to the JSONL file.
        skip_commented_out_lines (bool): If True, skip commented out lines.

    Returns:
        list[dict]: List of dictionaries parsed from the JSONL file.
    """
    result = []
    with open(file_path, "r") as f:
        if skip_commented_out_lines:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or line.startswith("//"):
                    continue
                result.append(json.loads(line))
        else:
            return [json.loads(line) for line in f]
    return result


def load_answers(path_to_answers: Path) -> Any:
    if path_to_answers.suffix == ".csv":
        return read_csv(path_to_answers)

    if path_to_answers.suffix == ".jsonl":
        return read_jsonl(str(path_to_answers))

    raise ValueError(f"Unsupported file format for answers: {path_to_answers}")


def get_runs_dir() -> Path:
    """Returns an absolute path to the directory storing runs."""

    return get_module_dir().parent / "runs"


def get_module_dir() -> Path:
    """Returns an absolute path to the MLE-bench module."""

    path = Path(__file__).parent.resolve()

    assert (
        path.name == "mlebench"
    ), f"Expected the module directory to be `mlebench`, but got `{path.name}`."

    return path


def get_repo_dir() -> Path:
    """Returns an absolute path to the repository directory."""

    return get_module_dir().parent


def generate_run_id(competition_id: str, agent_id: str, run_group: Optional[str] = None) -> str:
    """Creates a unique run ID for a specific competition and agent combo"""

    timestamp = get_timestamp()
    long_id = str(uuid.uuid4())
    short_id = long_id[:8]

    if run_group:  # the timestamp and agent are already included in the run group name
        return f"{competition_id}_{long_id}"

    return f"{timestamp}_{competition_id}_{agent_id}_{short_id}"


def create_run_dir(
    competition_id: Optional[str] = None,
    agent_id: Optional[str] = None,
    run_group: Optional[str] = None,
) -> Path:
    """Creates a directory for the run."""

    assert competition_id is None or isinstance(
        competition_id, str
    ), f"Expected a string or None, but got `{type(competition_id).__name__}`."

    assert agent_id is None or isinstance(
        agent_id, str
    ), f"Expected a string or None, but got `{type(agent_id).__name__}`."

    assert run_group is None or isinstance(
        run_group, str
    ), f"Expected a string or None, but got `{type(run_group).__name__}`."

    run_id = str(uuid.uuid4())

    if competition_id and agent_id:
        run_id = generate_run_id(competition_id, agent_id, run_group)

    run_dir = get_runs_dir() / run_id

    if run_group:
        run_dir = get_runs_dir() / run_group / run_id

    run_dir.mkdir(parents=True, exist_ok=False)

    assert isinstance(run_dir, Path), f"Expected a `Path`, but got `{type(run_dir)}`."
    assert run_dir.is_dir(), f"Expected a directory, but got `{run_dir}`."

    return run_dir


def is_compressed(fpath: Path) -> bool:
    """Checks if the file is compressed."""

    return fpath.suffix in [".zip", ".tar", ".gz", ".tgz", ".tar.gz", ".rar", ".7z"]


def compress(src: Path, compressed: Path, exist_ok: bool = False) -> None:
    """Compresses the contents of a source directory to a compressed file."""
    assert src.exists(), f"Source directory `{src}` does not exist."
    assert src.is_dir(), f"Expected a directory, but got `{src}`."
    if not exist_ok:
        assert not compressed.exists(), f"Compressed file `{compressed}` already exists."

    tqdm_desc = f"Compressing {src.name} to {compressed.name}"
    file_paths = [path for path in src.rglob("*") if path.is_file()]
    total_files = len(file_paths)

    def zip_compress(src: Path, compressed: Path):
        with zipfile.ZipFile(compressed, "w", zipfile.ZIP_DEFLATED) as zipf:
            for file_path in tqdm(file_paths, desc=tqdm_desc, unit="file", total=total_files):
                zipf.write(file_path, arcname=file_path.relative_to(src))

    def sevenz_compress(src: Path, compressed: Path):
        with py7zr.SevenZipFile(compressed, "w") as archive:
            for file_path in tqdm(file_paths, desc=tqdm_desc, unit="file", total=total_files):
                archive.write(file_path, arcname=file_path.relative_to(src))

    # Determine the compression format from the destination file suffix
    if compressed.suffix == ".zip":
        zip_compress(src, compressed)
    elif compressed.suffix == ".7z":
        sevenz_compress(src, compressed)
    else:
        raise NotImplementedError(f"Unsupported compression format: `{compressed.suffix}`.")


def extract(
    compressed: Path, dst: Path, recursive: bool = False, already_extracted: set = set()
) -> None:
    """Extracts the contents of a compressed file to a destination directory."""

    # pre-conditions
    assert compressed.exists(), f"File `{compressed}` does not exist."
    assert compressed.is_file(), f"Path `{compressed}` is not a file."
    assert is_compressed(compressed), f"File `{compressed}` is not compressed."

    if compressed.suffix == ".7z":
        with py7zr.SevenZipFile(compressed, mode="r") as ref:
            ref.extractall(dst)
    elif compressed.suffix == ".zip":
        with zipfile.ZipFile(compressed, "r") as ref:
            ref.extractall(dst)
    else:
        raise NotImplementedError(f"Unsupported compression format: `{compressed.suffix}`.")

    already_extracted.add(compressed)
    if recursive:
        to_extract = {
            fpath for fpath in set(dst.iterdir()) - already_extracted if is_compressed(fpath)
        }
        already_extracted.update(to_extract)

        for fpath in to_extract:
            extract(fpath, fpath.parent, recursive=True, already_extracted=already_extracted)


def is_empty(dir: Path) -> bool:
    """Checks if the directory is empty."""

    # pre-conditions
    assert isinstance(dir, Path), f"Expected a `Path`, but got `{type(dir)}`."
    assert dir.is_dir(), f"Expected a directory, but got `{dir}`."

    # body
    return not any(dir.iterdir())


def get_logger(name: str, level: int = logging.INFO, filename: Optional[Path] = None) -> Logger:
    logging.basicConfig(
        level=level,
        format="[%(asctime)s] [%(filename)s:%(lineno)d] %(message)s",
        filename=filename,
    )
    return logging.getLogger(name)


def load_yaml(fpath: Path) -> dict:
    """Loads a YAML file and returns its contents as a dictionary."""

    assert isinstance(fpath, Path), f"Expected a `Path`, but got `{type(fpath)}`."
    assert fpath.exists(), f"File `{fpath}` does not exist."
    assert fpath.is_file(), f"Expected a file, but got `{fpath}`."
    assert fpath.suffix == ".yaml", f"Expected a YAML file, but got `{fpath}`."

    with open(fpath, "r") as file:
        contents = yaml.safe_load(file)

    return contents


def in_ci() -> bool:
    """Checks if the code is running in GitHub CI."""

    return os.environ.get("GITHUB_ACTIONS") == "true"


def import_fn(fn_import_string: str) -> Callable:
    """
    Imports a function from a module given a string in the format
    `potentially.nested.module_name:fn_name`.

    Basically equivalent to `from potentially.nested.module_name import fn_name`.
    """
    module_name, fn_name = fn_import_string.split(":")
    module = importlib.import_module(module_name)
    fn = getattr(module, fn_name)
    return fn


def get_path_to_callable(callable: Callable) -> Path:
    """
    Retrieves the file path of the module where the given callable is defined.

    Args:
        callable (Callable): The callable for which the module path is required.

    Returns:
        Path: The relative path to the module file from the current working directory.

    Raises:
        AssertionError: If the module does not have a file path.
    """

    file_path = getattr(sys.modules[callable.__module__], "__file__", None)
    assert file_path is not None, f"Module `{callable.__module__}` does not have a file path."
    return Path(os.path.relpath(file_path, start=os.getcwd()))


def get_diff(d: dict, other_d: dict, fromfile: str = "d", tofile: str = "other_d") -> str:
    """Finds the differences between two nested dictionaries and returns a diff string."""

    d_str = yaml.dump(d, sort_keys=True)
    other_d_str = yaml.dump(other_d, sort_keys=True)

    diff = difflib.unified_diff(
        d_str.splitlines(),
        other_d_str.splitlines(),
        fromfile=fromfile,
        tofile=tofile,
        lineterm="",
    )

    return "\n".join(diff)


def read_csv(*args, **kwargs) -> DataFrame:
    """Reads a CSV file and returns a DataFrame with custom default kwargs."""

    try:
        new_default_kwargs = {"float_precision": "round_trip"}
        new_kwargs = {**new_default_kwargs, **kwargs}
        return pd.read_csv(*args, **new_kwargs)
    except pd.errors.EmptyDataError:
        logger.warning(f"CSV file empty! {args[0]}")
        return pd.DataFrame()


def get_timestamp() -> str:
    """Returns the current timestamp in the format `YYYY-MM-DDTHH-MM-SS-Z`."""

    return time.strftime("%Y-%m-%dT%H-%M-%S-%Z", time.gmtime())


logger = get_logger(__name__)
