"""Script for downloading kernels for competitions based on kernel references."""

import argparse

import kaggle
import nbformat
from nbconvert import PythonExporter
from tqdm.auto import tqdm

from mlebench.registry import registry
from mlebench.utils import get_logger

logger = get_logger(__name__)


def download_kernels(competition):
    """Downloads kernels for a given competition based on the kernel references."""

    kernel_refs_path = registry.get_competitions_dir() / competition.id / "kernels.txt"
    kernels_dir = registry.get_competitions_dir() / competition.id / "kernels"
    kernels_dir.mkdir(parents=True, exist_ok=True)

    kernel_refs = kernel_refs_path.read_text().splitlines()

    existing_kernels_py = []
    existing_kernels = []
    downloaded_kernels = []

    for kernel_ref in tqdm(
        kernel_refs,
        desc=f"Downloading kernels for {competition.id}",
        unit="kernel",
    ):
        author, kernel_slug = kernel_ref.split("/")
        py_script_path = kernels_dir / f"{author}_{kernel_slug}.py"
        notebook_path = kernels_dir / f"{author}_{kernel_slug}.ipynb"

        if notebook_path.exists() or py_script_path.exists():
            if notebook_path.exists():
                existing_kernels.append(kernel_ref)
            else:
                existing_kernels_py.append(kernel_ref)
            logger.debug(f"Kernel {kernel_ref} already exists. Skipping download.")
            continue
        else:
            try:
                kaggle.api.kernels_pull(kernel_ref, path=kernels_dir)
                downloaded_file = kernels_dir / f"{kernel_ref.split('/')[-1]}.ipynb"
                if downloaded_file.exists():
                    downloaded_kernels.append(kernel_ref)
                    new_filename = f"{author}_{downloaded_file.name}"
                    downloaded_file.rename(kernels_dir / new_filename)
            except kaggle.rest.ApiException as e:
                if e.status == 403:
                    logger.warning(
                        f"403 Forbidden error when downloading kernel {kernel_ref}. {e.body}. Skipping."
                    )
                else:
                    logger.error(f"Error downloading kernel {kernel_ref}: {str(e)}")

    total_kernels = len(kernel_refs)
    processed_kernels = len(downloaded_kernels + existing_kernels + existing_kernels_py)
    if processed_kernels != total_kernels:
        failed_kernels = set(kernel_refs) - set(
            downloaded_kernels + existing_kernels + existing_kernels_py
        )
        logger.warning(
            f"Failed to download {len(failed_kernels)} kernels for {competition.id}. "
            f"Failed kernels: {', '.join(failed_kernels)}"
        )

    for kernel_ref in tqdm(
        downloaded_kernels + existing_kernels,
        desc="Converting notebooks to Python scripts",
        unit="kernel",
    ):
        author, kernel_slug = kernel_ref.split("/")
        notebook_path = kernels_dir / f"{author}_{kernel_slug}.ipynb"
        py_script_path = kernels_dir / f"{author}_{kernel_slug}.py"
        try:
            ipynb_to_py(notebook_path, py_script_path)
            notebook_path.unlink()
        except Exception as e:
            logger.error(
                f"Error converting notebook {notebook_path} to Python script: {e}. Skipping."
            )
            if py_script_path.exists():
                py_script_path.unlink()  # Remove any partially converted file


def ipynb_to_py(path_to_ipynb, path_to_py):
    """Converts a Jupyter notebook to a Python script, removing comments."""

    def remove_comments_from_python_code(code):
        lines = code.split("\n")
        non_comment_lines = [line for line in lines if not line.strip().startswith("#")]
        return "\n".join(non_comment_lines)

    with open(path_to_ipynb, "r", encoding="utf-8") as ipynb_file:
        notebook_content = nbformat.read(ipynb_file, as_version=4)

    python_exporter = PythonExporter()
    python_content, _ = python_exporter.from_notebook_node(notebook_content)

    python_content = remove_comments_from_python_code(python_content)

    with open(path_to_py, "w", encoding="utf-8") as py_file:
        py_file.write(python_content)


parser = argparse.ArgumentParser(
    description="Download kernels for competitions from kernel references."
)

group = parser.add_mutually_exclusive_group(required=True)
group.add_argument(
    "-c",
    "--competition-id",
    help=(
        f"ID of the competition to download kernels for. "
        f"Valid options: {registry.list_competition_ids()}"
    ),
    type=str,
)
group.add_argument(
    "-a",
    "--all",
    help="Download kernels for all competitions.",
    action="store_true",
)
group.add_argument(
    "-l",
    "--list",
    help="Path to a text file listing competition IDs, one per line.",
    type=str,
)

args = parser.parse_args()

if args.all:
    competition_ids = registry.list_competition_ids()
elif args.list:
    with open(args.list, "r") as f:
        competition_ids = f.read().splitlines()
else:
    competition_ids = [args.competition_id]

competitions = [registry.get_competition(cid) for cid in competition_ids]

for competition in competitions:
    download_kernels(competition)
