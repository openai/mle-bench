"""
Tests for the research tasks functionality.
"""

import os
import tempfile
from pathlib import Path

import pytest

from researchbench.research_tasks.manager import (
    list_research_tasks,
    get_research_task,
    prepare_research_task,
    run_research_task,
)


def test_list_research_tasks():
    """Test listing research tasks."""
    tasks = list_research_tasks()
    assert len(tasks) > 0
    assert isinstance(tasks, list)
    assert all(isinstance(task, dict) for task in tasks)
    assert all("id" in task for task in tasks)
    assert all("name" in task for task in tasks)
    assert all("category" in task for task in tasks)


def test_list_research_tasks_by_category():
    """Test listing research tasks by category."""
    tasks = list_research_tasks(category="literature-review")
    assert len(tasks) > 0
    assert all(task["category"] == "literature-review" for task in tasks)


def test_get_research_task():
    """Test getting a research task by ID."""
    task = get_research_task("literature-review-ai-ethics")
    assert task["id"] == "literature-review-ai-ethics"
    assert task["category"] == "literature-review"
    assert "name" in task
    assert "description" in task


def test_prepare_research_task():
    """Test preparing a research task."""
    with tempfile.TemporaryDirectory() as temp_dir:
        data_dir = Path(temp_dir)
        prepare_research_task("literature-review-ai-ethics", data_dir)
        
        # Check that the task directory was created
        task_dir = data_dir / "literature-review-ai-ethics"
        assert task_dir.exists()
        assert task_dir.is_dir()
        
        # Check that the metadata file was created
        metadata_file = task_dir / "metadata.json"
        assert metadata_file.exists()
        assert metadata_file.is_file()
        
        # Check that the README file was created
        readme_file = task_dir / "README.md"
        assert readme_file.exists()
        assert readme_file.is_file()


def test_run_research_task():
    """Test running a research task."""
    with tempfile.TemporaryDirectory() as temp_dir:
        data_dir = Path(temp_dir) / "data"
        output_dir = Path(temp_dir) / "output"
        data_dir.mkdir()
        output_dir.mkdir()
        
        # Prepare the task
        prepare_research_task("literature-review-ai-ethics", data_dir)
        
        # Run the task
        result = run_research_task("literature-review-ai-ethics", data_dir, output_dir, {})
        
        # Check the result
        assert "task_id" in result
        assert result["task_id"] == "literature-review-ai-ethics"
        assert "status" in result
        assert result["status"] == "completed"
        assert "output_files" in result
        assert len(result["output_files"]) > 0
        
        # Check that the output files were created
        task_output_dir = output_dir / "literature-review-ai-ethics"
        assert task_output_dir.exists()
        assert task_output_dir.is_dir()
        
        for output_file in result["output_files"]:
            assert (task_output_dir / output_file).exists()
            assert (task_output_dir / output_file).is_file()


if __name__ == "__main__":
    pytest.main(["-xvs", __file__])