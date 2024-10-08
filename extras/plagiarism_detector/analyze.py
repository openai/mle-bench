import concurrent.futures
import json
import subprocess
import time
from pathlib import Path
from subprocess import TimeoutExpired

from tqdm import tqdm

from mlebench.registry import registry
from mlebench.utils import get_logger, read_jsonl

logger = get_logger(__name__)


def is_agent_submission(file_path: str, agent_submission_dir: Path) -> bool:
    """
    Check if a code file is part of an agent's submission.
    """
    return agent_submission_dir.as_posix() in file_path


def process_dolos_report(report: list, agent_submission_dir: Path) -> list:
    """
    Filter the full dolos report to include only comparisons between files in the agent's submission directory and kernel files.
    Sort the filtered results by similarity score in descending order.
    """
    plagiarism_report = []
    for item in report:
        left_file = item["leftFile"]
        right_file = item["rightFile"]
        if (
            is_agent_submission(left_file, agent_submission_dir)
            and not is_agent_submission(right_file, agent_submission_dir)
        ) or (
            is_agent_submission(right_file, agent_submission_dir)
            and not is_agent_submission(left_file, agent_submission_dir)
        ):
            plagiarism_report.append(item)

    sorted_report = sorted(plagiarism_report, key=lambda x: x["similarity"], reverse=True)

    return sorted_report


def process_submission(submission, dolos_dir, timeout=500) -> tuple[str, list]:
    """
    Process a single submission for plagiarism detection.

    Args:
        submission (dict): A dictionary containing submission details, including 'competition_id' and 'code_path'.
        dolos_dir (str): The directory containing the dolos script.
        timeout (int): Timeout for the plagiarism detection process in seconds.

    Returns:
        tuple[str, list]: A tuple containing the code path and a list of plagiarism report items.
    """
    competition_id = submission["competition_id"]
    code_path = submission.get("code_path")

    if code_path is None:
        logger.warning(f"No code_path found for submission in competition {competition_id}")
        return None, None

    kernels_dir = registry.get_competitions_dir() / competition_id / "kernels"

    if not kernels_dir.exists() or not any(f.suffix == ".py" for f in kernels_dir.iterdir()):
        raise ValueError(
            f"Kernels for competition {competition_id} not found. Run `mlebench download-kernels -c {competition_id}` or `mlebench download-kernels --all` to download."
        )

    kernel_files = [str(f.resolve()) for f in kernels_dir.glob("*.py")]
    assert len(kernel_files) > 0, f"No kernel files found for competition {competition_id}"

    code_path = Path(code_path).resolve()

    if code_path.is_dir():
        agent_files = list(code_path.rglob("*.py"))
        logger.info(f"Found {len(agent_files)} Python files in {code_path}")
    else:
        agent_files = [code_path] if code_path.suffix == ".py" else []

    assert agent_files, f"No Python files found for submission {code_path}"

    agent_files = [str(f) for f in agent_files]
    logger.info(f"Total agent files to process: {len(agent_files)}")

    if not Path(code_path).exists():
        logger.warning(f"Submission file not found: {code_path}")
        return

    # Combine all files for comparison
    all_files = kernel_files + agent_files

    logger.info(f"Comparing {len(all_files)} files")
    logger.info(f"Agent files: {agent_files}")
    logger.info(f"Kernel files: {kernel_files}")

    # Run the plagiarism detector with a timeout
    cmd = ["node", "dolos_wrapper.mjs", "--files"] + all_files
    start_time = time.time()
    try:
        result = subprocess.run(cmd, cwd=dolos_dir, capture_output=True, text=True, timeout=timeout)

        if result.returncode == 0:
            logger.info(f"Plagiarism detection completed successfully for {code_path}")
            try:
                dolos_report = json.loads(result.stdout)
                plagiarism_report = process_dolos_report(dolos_report, Path(code_path).parent)
                return code_path, plagiarism_report
            except json.JSONDecodeError:
                logger.error(f"Error parsing plagiarism detection output for {code_path}")
                logger.error(result.stdout)
        else:
            logger.error(f"Error running plagiarism detection for {code_path}:")
            logger.error(result.stdout)
            logger.error(result.stderr)

    except TimeoutExpired:
        elapsed_time = time.time() - start_time
        logger.warning(
            f"Plagiarism detection timed out after {elapsed_time:.2f} seconds for {code_path}"
        )

    return None, None


def run_plagiarism_detector(
    submission: Path,
    output_dir: Path,
    use_concurrency: bool = True,
    timeout: int = 500,
):
    """
    Run the plagiarism detector on all submissions, optionally using concurrency.

    Save the plagiarism report for each competition to a JSON file.

    Args:
        submission (str): Path to the submission JSONL file.
        output_dir (str): Path to the output directory.
        concurrent (bool): Whether to run the plagiarism detector concurrently.
    """
    script_dir = Path(__file__).parent.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    submissions = read_jsonl(submission)

    results = []

    if use_concurrency:
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = [
                executor.submit(process_submission, submission, script_dir, timeout)
                for submission in submissions
            ]

            for future in tqdm(
                concurrent.futures.as_completed(futures),
                total=len(submissions),
                desc="Processing submissions",
            ):
                code_path, plagiarism_report = future.result()
                if code_path and plagiarism_report:
                    results.append((code_path, plagiarism_report))
    else:
        for submission in tqdm(submissions, desc="Processing submissions"):
            code_path, plagiarism_report = process_submission(submission, script_dir, timeout)
            if code_path and plagiarism_report:
                results.append((code_path, plagiarism_report))

    for code_path, plagiarism_report in results:
        formatted_path = (
            str(code_path).replace("/", "_").replace("\\", "_")
        )  # includes competition_id and run_id in the filename
        output_file = output_dir / f"{formatted_path}_plagiarism_report.jsonl"
        with open(output_file, "w") as f:
            for item in plagiarism_report:
                json.dump(item, f)
                f.write("\n")
        logger.info(f"Plagiarism report saved as {output_file}")
