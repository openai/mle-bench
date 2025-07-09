import json
import os
import shutil
from pathlib import Path
from typing import Dict, List, Any, Optional, Callable

from researchbench.utils import get_logger

logger = get_logger(__name__)

# Define research task categories
CATEGORIES = [
    "literature-review",
    "data-analysis",
    "hypothesis-testing",
    "survey-design",
    "qualitative-research",
    "meta-analysis",
    "experimental-design",
    "natural-language-processing",
    "computer-vision",
    "reinforcement-learning",
    "social-science",
    "economics",
    "psychology",
    "biology",
    "physics",
    "chemistry",
    "medicine",
    "environmental-science",
]

# Sample research tasks
RESEARCH_TASKS = [
    {
        "id": "literature-review-ai-ethics",
        "name": "Literature Review on AI Ethics",
        "description": "Conduct a comprehensive literature review on ethical considerations in AI development and deployment.",
        "category": "literature-review",
        "data_requirements": ["papers-ai-ethics"],
        "expected_output": "A structured literature review with key themes, gaps, and future research directions.",
    },
    {
        "id": "data-analysis-climate-change",
        "name": "Climate Change Data Analysis",
        "description": "Analyze global temperature and CO2 emission data to identify trends and correlations.",
        "category": "data-analysis",
        "data_requirements": ["global-temperature-data", "co2-emissions-data"],
        "expected_output": "Statistical analysis report with visualizations and insights.",
    },
    {
        "id": "nlp-sentiment-analysis",
        "name": "Sentiment Analysis of Social Media Posts",
        "description": "Develop a sentiment analysis model to analyze social media posts about a specific topic.",
        "category": "natural-language-processing",
        "data_requirements": ["social-media-dataset"],
        "expected_output": "A trained sentiment analysis model with evaluation metrics and analysis report.",
    },
    {
        "id": "survey-design-remote-work",
        "name": "Survey Design for Remote Work Research",
        "description": "Design a comprehensive survey to study the impact of remote work on productivity and well-being.",
        "category": "survey-design",
        "data_requirements": [],
        "expected_output": "A well-structured survey with question types, scales, and sampling strategy.",
    },
    {
        "id": "experimental-design-psychology",
        "name": "Experimental Design for Cognitive Psychology Study",
        "description": "Design an experiment to test working memory capacity under different conditions.",
        "category": "experimental-design",
        "data_requirements": [],
        "expected_output": "A detailed experimental protocol with variables, controls, and analysis plan.",
    },
]

# Registry of task handlers
TASK_HANDLERS: Dict[str, Callable] = {}


def register_task_handler(task_id: str):
    """Decorator to register a task handler function."""
    def decorator(func: Callable):
        TASK_HANDLERS[task_id] = func
        return func
    return decorator


def list_research_tasks(category: Optional[str] = None) -> List[Dict[str, Any]]:
    """List available research tasks, optionally filtered by category."""
    if category:
        return [task for task in RESEARCH_TASKS if task["category"] == category]
    return RESEARCH_TASKS


def get_research_task(task_id: str) -> Dict[str, Any]:
    """Get a research task by ID."""
    for task in RESEARCH_TASKS:
        if task["id"] == task_id:
            return task
    raise ValueError(f"Research task with ID '{task_id}' not found.")


def prepare_research_task(task_id: str, data_dir: Path) -> None:
    """Prepare a research task by downloading and setting up required data."""
    task = get_research_task(task_id)
    
    # Create task directory
    task_dir = data_dir / task_id
    task_dir.mkdir(parents=True, exist_ok=True)
    
    # Save task metadata
    with open(task_dir / "metadata.json", "w") as f:
        json.dump(task, f, indent=2)
    
    # Download required datasets
    for dataset_id in task.get("data_requirements", []):
        download_dataset(dataset_id, task_dir)
    
    # Create task-specific files
    create_task_files(task, task_dir)


def download_dataset(dataset_id: str, task_dir: Path) -> None:
    """Download a dataset required for a research task."""
    dataset_dir = task_dir / "datasets" / dataset_id
    dataset_dir.mkdir(parents=True, exist_ok=True)
    
    # This is a placeholder. In a real implementation, this would download
    # the actual dataset from a repository or API.
    logger.info(f"Downloading dataset: {dataset_id}")
    
    # Create a sample dataset file for demonstration
    with open(dataset_dir / "README.md", "w") as f:
        f.write(f"# {dataset_id}\n\nThis is a placeholder for the {dataset_id} dataset.")


def create_task_files(task: Dict[str, Any], task_dir: Path) -> None:
    """Create task-specific files based on the task type."""
    # Create a README file with task instructions
    with open(task_dir / "README.md", "w") as f:
        f.write(f"# {task['name']}\n\n")
        f.write(f"{task['description']}\n\n")
        f.write("## Expected Output\n\n")
        f.write(f"{task['expected_output']}\n\n")
        f.write("## Data Requirements\n\n")
        for dataset_id in task.get("data_requirements", []):
            f.write(f"- {dataset_id}\n")
    
    # Create a template for the output
    output_template_dir = task_dir / "templates"
    output_template_dir.mkdir(exist_ok=True)
    
    with open(output_template_dir / "output_template.md", "w") as f:
        f.write(f"# {task['name']} - Results\n\n")
        f.write("## Summary\n\n[Provide a summary of your findings here]\n\n")
        f.write("## Methodology\n\n[Describe your methodology here]\n\n")
        f.write("## Results\n\n[Present your results here]\n\n")
        f.write("## Discussion\n\n[Discuss your findings here]\n\n")
        f.write("## Conclusion\n\n[Provide your conclusions here]\n\n")


def run_research_task(task_id: str, data_dir: Path, output_dir: Path, params: Dict[str, Any]) -> Dict[str, Any]:
    """Run a research task with the given parameters."""
    task = get_research_task(task_id)
    
    # Create output directory
    task_output_dir = output_dir / task_id
    task_output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save parameters
    with open(task_output_dir / "params.json", "w") as f:
        json.dump(params, f, indent=2)
    
    # Check if there's a registered handler for this task
    if task_id in TASK_HANDLERS:
        result = TASK_HANDLERS[task_id](data_dir / task_id, task_output_dir, params)
    else:
        # Default handler
        result = default_task_handler(task, data_dir / task_id, task_output_dir, params)
    
    # Save result
    with open(task_output_dir / "result.json", "w") as f:
        json.dump(result, f, indent=2)
    
    return result


def default_task_handler(
    task: Dict[str, Any], 
    task_dir: Path, 
    output_dir: Path, 
    params: Dict[str, Any]
) -> Dict[str, Any]:
    """Default handler for research tasks without a specific implementation."""
    # Copy template to output directory
    template_path = task_dir / "templates" / "output_template.md"
    if template_path.exists():
        shutil.copy(template_path, output_dir / "output.md")
    
    # Return a basic result structure
    return {
        "task_id": task["id"],
        "status": "completed",
        "message": "Task completed with default handler. Please implement a specific handler for better results.",
        "output_files": ["output.md"],
    }


# Example task handler registration
@register_task_handler("literature-review-ai-ethics")
def handle_literature_review_ai_ethics(task_dir: Path, output_dir: Path, params: Dict[str, Any]) -> Dict[str, Any]:
    """Handler for the AI ethics literature review task."""
    # In a real implementation, this would perform the actual literature review
    # or provide tools and guidance for the user to conduct it.
    
    # Create a sample output file
    with open(output_dir / "literature_review.md", "w") as f:
        f.write("# Literature Review on AI Ethics\n\n")
        f.write("## Introduction\n\n")
        f.write("This literature review examines the ethical considerations in AI development and deployment.\n\n")
        f.write("## Key Themes\n\n")
        f.write("1. Fairness and Bias\n")
        f.write("2. Transparency and Explainability\n")
        f.write("3. Privacy and Data Protection\n")
        f.write("4. Accountability and Responsibility\n")
        f.write("5. Human Autonomy and Agency\n\n")
        f.write("## Research Gaps\n\n")
        f.write("Based on the literature, the following research gaps have been identified...\n\n")
        f.write("## Future Research Directions\n\n")
        f.write("Future research should focus on...\n\n")
    
    return {
        "task_id": "literature-review-ai-ethics",
        "status": "completed",
        "message": "Literature review completed successfully.",
        "output_files": ["literature_review.md"],
        "key_themes": [
            "Fairness and Bias",
            "Transparency and Explainability",
            "Privacy and Data Protection",
            "Accountability and Responsibility",
            "Human Autonomy and Agency",
        ],
    }


@register_task_handler("data-analysis-climate-change")
def handle_climate_change_data_analysis(task_dir: Path, output_dir: Path, params: Dict[str, Any]) -> Dict[str, Any]:
    """Handler for the climate change data analysis task."""
    # In a real implementation, this would perform actual data analysis
    # or provide tools and guidance for the user to conduct it.
    
    # Create a sample output file
    with open(output_dir / "climate_analysis.md", "w") as f:
        f.write("# Climate Change Data Analysis\n\n")
        f.write("## Methodology\n\n")
        f.write("This analysis examines global temperature and CO2 emission data to identify trends and correlations.\n\n")
        f.write("## Key Findings\n\n")
        f.write("1. Global temperatures have increased by X°C over the past century.\n")
        f.write("2. CO2 emissions show a strong correlation with temperature increases (r=0.85).\n")
        f.write("3. Regional variations in warming patterns are significant.\n\n")
        f.write("## Visualizations\n\n")
        f.write("(Visualizations would be included here in a real implementation)\n\n")
        f.write("## Conclusions\n\n")
        f.write("The data strongly supports the conclusion that...\n\n")
    
    return {
        "task_id": "data-analysis-climate-change",
        "status": "completed",
        "message": "Climate change data analysis completed successfully.",
        "output_files": ["climate_analysis.md"],
        "key_findings": [
            "Global temperatures have increased by X°C over the past century.",
            "CO2 emissions show a strong correlation with temperature increases (r=0.85).",
            "Regional variations in warming patterns are significant.",
        ],
    }