"""
Contains functions to manually generate a textual preview of some common file types (.csv, .json,..) for the agent.
"""

import json
import os
import logging
from pathlib import Path

import humanize
import pandas as pd
from genson import SchemaBuilder
from pandas.api.types import is_numeric_dtype

# these files are treated as code (e.g. markdown wrapped)
code_files = {".py", ".sh", ".yaml", ".yml", ".md", ".html", ".xml", ".log", ".rst"}
# we treat these files as text (rather than binary) files
plaintext_files = {".txt", ".csv", ".json", ".tsv"} | code_files


def get_file_len_size(f: Path) -> tuple[int, str]:
    """
    Calculate the size of a file (#lines for plaintext files, otherwise #bytes)
    Also returns a human-readable string representation of the size.
    """
    if f.suffix in plaintext_files:
        num_lines = sum(1 for _ in open(f))
        return num_lines, f"{num_lines} lines"
    else:
        s = f.stat().st_size
        return s, humanize.naturalsize(s)


def file_tree(path: Path, depth=0) -> str:
    result = []
    files = [p for p in Path(path).iterdir() if not p.is_dir()]
    dirs = [p for p in Path(path).iterdir() if p.is_dir()]

    if depth in [0, 1]:
        max_n = 15
    else:
        max_n = 5 if len(files) > 30 else 8

    for p in sorted(files)[:max_n]:
        result.append(f"{' '*depth*4}{p.name} ({get_file_len_size(p)[1]})")
    if len(files) > max_n:
        result.append(f"{' '*depth*4}... and {len(files)-max_n} other files")

    for p in sorted(dirs):
        result.append(f"{' '*depth*4}{p.name}/")
        result.append(file_tree(p, depth + 1))

    return "\n".join(result)


def _walk(path: Path):
    for p in sorted(Path(path).iterdir()):
        if p.is_dir():
            yield from _walk(p)
            continue
        yield p


def preview_csv(p: Path, file_name: str, simple=True) -> str:
    """Generate a textual preview of a csv file

    Args:
        p (Path): the path to the csv file
        file_name (str): the file name to use in the preview
        simple (bool, optional): whether to use a simplified version of the preview. Defaults to True.

    Returns:
        str: the textual preview
    """
    df = pd.read_csv(p)

    out = []
    
    # --- 修改重点：强化行数意识 ---
    row_count = df.shape[0]
    out.append(f"-> {file_name} has {row_count} rows and {df.shape[1]} columns.")
    
    if "test" in file_name.lower():
        out.append(f"⚠️  CRITICAL: This is the TEST set. You MUST predict for all {row_count} rows. DO NOT truncate or sample this data.")
    
    if "sample_submission" in file_name.lower() or "submission" in file_name.lower():
        out.append(f"⚠️  IMPORTANT: The submission MUST have EXACTLY {row_count} rows matching the test set IDs.")
        out.append(f"The exact column names are: {', '.join(df.columns.tolist())}")
    if "sample_submission" in file_name.lower() or "submission" in file_name.lower():
        out.append("⚠️  IMPORTANT: This is the CORRECT submission format that must be followed!")
        out.append(f"The exact column names are: {', '.join(df.columns.tolist())}")
        out.append("Any description.md format information should be IGNORED if it conflicts with this file.")
        
    if simple:
        cols = df.columns.tolist()
        sel_cols = 15
        cols_str = ", ".join(cols[:sel_cols])
        res = f"The columns are: {cols_str}"
        if len(cols) > sel_cols:
            res += f"... and {len(cols)-sel_cols} more columns"
        out.append(res)
    else:
        out.append("Here is some information about the columns:")
        for col in sorted(df.columns):
            dtype = df[col].dtype
            name = f"{col} ({dtype})"

            nan_count = df[col].isnull().sum()

            if dtype == "bool":
                v = df[col][df[col].notnull()].mean()
                out.append(f"{name} is {v*100:.2f}% True, {100-v*100:.2f}% False")
            elif df[col].nunique() < 10:
                out.append(
                    f"{name} has {df[col].nunique()} unique values: {df[col].unique().tolist()}"
                )
            elif is_numeric_dtype(df[col]):
                out.append(
                    f"{name} has range: {df[col].min():.2f} - {df[col].max():.2f}, {nan_count} nan values"
                )
            elif dtype == "object":
                out.append(
                    f"{name} has {df[col].nunique()} unique values. Some example values: {df[col].value_counts().head(4).index.tolist()}"
                )

    return "\n".join(out)


def preview_json(p: Path, file_name: str):
    builder = SchemaBuilder()
    with open(p) as f:
        first_line = f.readline().strip()

        try:
            first_object = json.loads(first_line)

            if not isinstance(first_object, dict):
                raise json.JSONDecodeError("The first line isn't JSON", first_line, 0)

            # if the the next line exists and is not empty, then it is a JSONL file
            second_line = f.readline().strip()
            if second_line:
                f.seek(0)  # so reset and read line by line
                for line in f:
                    builder.add_object(json.loads(line.strip()))
            # if it is empty, then it's a single JSON object file
            else:
                builder.add_object(first_object)

        except json.JSONDecodeError:
            # if first line isn't JSON, then it's prettified and we can read whole file
            f.seek(0)
            builder.add_object(json.load(f))

    return f"-> {file_name} has auto-generated json schema:\n" + builder.to_json(
        indent=2
    )


def generate(base_path, include_file_details=True, simple=False):
    """Generate a textual preview of a directory (structure + file previews)."""
    tree = f"```\n{file_tree(base_path)}```"
    out = [tree]

    if include_file_details:
        for fn in _walk(base_path):
            file_name = str(fn.relative_to(base_path))

            if fn.suffix == ".csv":
                out.append(preview_csv(fn, file_name, simple=simple))
            elif fn.suffix == ".json":
                out.append(preview_json(fn, file_name))
            elif fn.suffix in plaintext_files:
                if get_file_len_size(fn)[0] < 30:
                    with open(fn) as f:
                        content = f.read()
                        if fn.suffix in code_files:
                            content = f"```\n{content}\n```"
                        out.append(f"-> {file_name} has content:\n\n{content}")

    base_path_obj = Path(base_path)
    input_dir = base_path_obj / "input"

    if input_dir.exists():
        has_validation = any(
            item.name.lower().startswith('val') or 'validation' in item.name.lower()
            for item in input_dir.iterdir()
        )

        if has_validation:
            msg = []
            msg.append("\n**COMPETITION DATA STRATEGY - I will READ CAREFULLY**")
            msg.append(
                "\n"
                "In competitions, 'validation' files are NOT always unlabeled test data.\n"
                "They often contain labels and should be treated as additional training data.\n"
                "\n"
                "REQUIRED STEPS:\n"
                "1. I need check if validation files contain labels (inspect file structure)\n"
                "2. **If labels exist → I will merge train + validation into one dataset. I will confirm that the original val data has been merged into the training set.**\n"
                "3. I will Create my own train/val split from the merged data\n"
                "\n"
                "This approach maximizes training data and often improves performance.\n"
                "**Note**: If existing code already implements this strategy, i will skip this step.\n"
            )
            
            out.append("\n".join(msg))
            
    result = "\n\n".join(out)

    # if the result is very long we generate a simpler version
    if len(result) > 6_000 and not simple:
        return generate(
            base_path, include_file_details=include_file_details, simple=True
        )
    # if still too long, we truncate
    if len(result) > 6_000 and simple:
        return result[:6_000] + "\n... (truncated)"

    return result


logger = logging.getLogger("aide")


def clean_task_desc(task_desc: str, cfg) -> str:
    """Clean task_desc with LLM (remove env-only noise, keep core task); append sample_submission format if present."""
    from aide.backend import query

    acfg = cfg.agent

    prompt = {
        "Task": "Remove ONLY useless environment information from the task description below. Keep all core task content.",
        "Instructions": [
            "**What to REMOVE** (not related to the essence of the task):",
            "  • Internet access restrictions (e.g., 'Internet access: disabled', 'no internet')",
            "  • URLs and web links (e.g., 'https://...')",
            "  • Time/execution limits (e.g., 'time limit: 3 or xx hours', '≤ xx hours')",
            "  • Notebook-related mentions (e.g., 'Jupyter notebook', '.ipynb')",
            "  • Rankings, bonuses, and other irrelevant information",
            "",
            "**What to KEEP** (preserve exactly):",
            "  • Task goal and requirements",
            "  • Dataset description and data format",
            "  • Evaluation metric definition",
            "  • Submission format requirements",
            "  • All other task-specific information",
            "",
            "**Output**: Return ONLY the cleaned task description text, no explanations."
        ],
        "Task Description to Clean": task_desc
    }

    try:
        cleaned_desc = query(
            system_message=prompt,
            user_message=None,
            model=acfg.cheap.model,
            temperature=1.0,
            cfg=cfg
        )
        logger.info(f"Task description cleaned for code review")
        cleaned_desc = cleaned_desc.strip()
    except Exception as e:
        logger.warning(f"Failed to clean task_desc with LLM: {e}. Using original.")
        cleaned_desc = task_desc

    input_dir = os.path.join(cfg.workspace_dir, "input")
    sample_submission_paths = [
        os.path.join(input_dir, "sample_submission.csv"),
        os.path.join(input_dir, "sampleSubmission.csv")
    ]

    for sample_path in sample_submission_paths:
        if os.path.exists(sample_path):
            try:
                # 获取真实的总行数
                full_df = pd.read_csv(sample_path)
                total_rows = len(full_df)
                example_df = full_df.head(5)

                submission_format = "\n\n" + "=" * 60 + "\n"
                submission_format += "**STRICT SUBMISSION INTEGRITY REQUIREMENTS**\n"
                submission_format += "=" * 60 + "\n"
                submission_format += f"1. TOTAL ROWS REQUIRED: {total_rows}\n"
                submission_format += f"2. ALL IDs MUST BE PRESENT: Your submission must contain every ID from the test set.\n"
                submission_format += "3. NO TRUNCATION: Do not use .head(), .sample(), or nrows during final inference.\n\n"
                submission_format += "FORMAT EXAMPLE (First 5 rows):\n"
                submission_format += example_df.to_string(index=False)
                submission_format += "\n" + "=" * 60 + "\n"
                
                # 显式提示模型写 assert 语句
                submission_format += "\n**IMPLEMENTATION HINT**: Always include `assert len(submission) == {total_rows}` in your inference script.\n"

                cleaned_desc += submission_format
                logger.info(f"Added submission format example from {os.path.basename(sample_path)}")
                break
            except Exception as e:
                logger.warning(f"Failed to read {sample_path}: {e}")
                continue
    logger.info(f"Generating Task desc: \n  {cleaned_desc} \n")
    return cleaned_desc