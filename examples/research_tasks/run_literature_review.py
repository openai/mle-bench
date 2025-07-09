#!/usr/bin/env python
"""
Example script to run a literature review research task.
"""

import argparse
import json
from pathlib import Path

from researchbench.research_tasks.manager import get_research_task, prepare_research_task, run_research_task


def main():
    parser = argparse.ArgumentParser(description="Run a literature review research task.")
    parser.add_argument(
        "--task-id",
        type=str,
        default="literature-review-ai-ethics",
        help="ID of the research task to run.",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=str(Path.home() / ".researchbench" / "data"),
        help="Path to the directory where the data is stored.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(Path.home() / ".researchbench" / "output"),
        help="Path to the directory where the output will be stored.",
    )
    parser.add_argument(
        "--params",
        type=str,
        default="{}",
        help="JSON string with parameters for the research task.",
    )
    args = parser.parse_args()

    # Convert paths to Path objects
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    
    # Parse parameters
    params = json.loads(args.params)
    
    # Get task information
    task = get_research_task(args.task_id)
    print(f"Running research task: {task['name']}")
    print(f"Description: {task['description']}")
    print(f"Category: {task['category']}")
    
    # Prepare the task
    print(f"Preparing task {args.task_id}...")
    prepare_research_task(args.task_id, data_dir)
    
    # Run the task
    print(f"Running task {args.task_id}...")
    result = run_research_task(args.task_id, data_dir, output_dir, params)
    
    # Print the result
    print("\nTask completed successfully!")
    print(f"Output files: {', '.join(result.get('output_files', []))}")
    print(f"Output directory: {output_dir / args.task_id}")
    
    # Print key findings if available
    if "key_themes" in result:
        print("\nKey themes identified:")
        for theme in result["key_themes"]:
            print(f"- {theme}")
    
    if "research_gaps" in result:
        print("\nResearch gaps identified:")
        for gap in result["research_gaps"]:
            print(f"- {gap}")
    
    if "future_directions" in result:
        print("\nFuture research directions:")
        for direction in result["future_directions"]:
            print(f"- {direction}")


if __name__ == "__main__":
    main()