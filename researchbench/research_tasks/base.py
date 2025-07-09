from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Any, List, Optional


class ResearchTask(ABC):
    """Base class for all research tasks."""

    def __init__(self, task_id: str, name: str, description: str, category: str):
        self.task_id = task_id
        self.name = name
        self.description = description
        self.category = category
        self.data_requirements: List[str] = []
        self.expected_output: str = ""

    @abstractmethod
    def prepare(self, data_dir: Path) -> None:
        """Prepare the task by downloading and setting up required data."""
        pass

    @abstractmethod
    def run(self, data_dir: Path, output_dir: Path, params: Dict[str, Any]) -> Dict[str, Any]:
        """Run the task with the given parameters."""
        pass

    def get_metadata(self) -> Dict[str, Any]:
        """Get task metadata."""
        return {
            "id": self.task_id,
            "name": self.name,
            "description": self.description,
            "category": self.category,
            "data_requirements": self.data_requirements,
            "expected_output": self.expected_output,
        }


class LiteratureReviewTask(ResearchTask):
    """Base class for literature review tasks."""

    def __init__(self, task_id: str, name: str, description: str, topic: str):
        super().__init__(task_id, name, description, "literature-review")
        self.topic = topic
        self.expected_output = "A structured literature review with key themes, gaps, and future research directions."

    def prepare(self, data_dir: Path) -> None:
        """Prepare the literature review task."""
        task_dir = data_dir / self.task_id
        task_dir.mkdir(parents=True, exist_ok=True)
        
        # Create a README file with task instructions
        with open(task_dir / "README.md", "w") as f:
            f.write(f"# {self.name}\n\n")
            f.write(f"{self.description}\n\n")
            f.write(f"## Topic: {self.topic}\n\n")
            f.write("## Expected Output\n\n")
            f.write(f"{self.expected_output}\n\n")
        
        # Create a template for the literature review
        template_dir = task_dir / "templates"
        template_dir.mkdir(exist_ok=True)
        
        with open(template_dir / "literature_review_template.md", "w") as f:
            f.write(f"# Literature Review: {self.topic}\n\n")
            f.write("## Introduction\n\n[Provide an introduction to the topic and the scope of the review]\n\n")
            f.write("## Methodology\n\n[Describe how you conducted the literature search and selection]\n\n")
            f.write("## Key Themes\n\n[Identify and discuss the main themes in the literature]\n\n")
            f.write("## Research Gaps\n\n[Identify gaps in the current research]\n\n")
            f.write("## Future Research Directions\n\n[Suggest directions for future research]\n\n")
            f.write("## Conclusion\n\n[Summarize the main findings of the review]\n\n")
            f.write("## References\n\n[List all references cited in the review]\n\n")


class DataAnalysisTask(ResearchTask):
    """Base class for data analysis tasks."""

    def __init__(self, task_id: str, name: str, description: str, dataset_ids: List[str]):
        super().__init__(task_id, name, description, "data-analysis")
        self.data_requirements = dataset_ids
        self.expected_output = "Statistical analysis report with visualizations and insights."

    def prepare(self, data_dir: Path) -> None:
        """Prepare the data analysis task."""
        task_dir = data_dir / self.task_id
        task_dir.mkdir(parents=True, exist_ok=True)
        
        # Create a README file with task instructions
        with open(task_dir / "README.md", "w") as f:
            f.write(f"# {self.name}\n\n")
            f.write(f"{self.description}\n\n")
            f.write("## Datasets\n\n")
            for dataset_id in self.data_requirements:
                f.write(f"- {dataset_id}\n")
            f.write("\n## Expected Output\n\n")
            f.write(f"{self.expected_output}\n\n")
        
        # Create a template for the data analysis report
        template_dir = task_dir / "templates"
        template_dir.mkdir(exist_ok=True)
        
        with open(template_dir / "data_analysis_template.md", "w") as f:
            f.write(f"# Data Analysis Report: {self.name}\n\n")
            f.write("## Introduction\n\n[Provide an introduction to the analysis and its objectives]\n\n")
            f.write("## Data Description\n\n[Describe the datasets used in the analysis]\n\n")
            f.write("## Methodology\n\n[Describe the analytical methods used]\n\n")
            f.write("## Exploratory Data Analysis\n\n[Present initial data exploration and visualizations]\n\n")
            f.write("## Statistical Analysis\n\n[Present statistical tests and models]\n\n")
            f.write("## Results\n\n[Present the main findings of the analysis]\n\n")
            f.write("## Discussion\n\n[Interpret the results and discuss their implications]\n\n")
            f.write("## Conclusion\n\n[Summarize the main findings and their significance]\n\n")


class ExperimentalDesignTask(ResearchTask):
    """Base class for experimental design tasks."""

    def __init__(self, task_id: str, name: str, description: str, field: str):
        super().__init__(task_id, name, description, "experimental-design")
        self.field = field
        self.expected_output = "A detailed experimental protocol with variables, controls, and analysis plan."

    def prepare(self, data_dir: Path) -> None:
        """Prepare the experimental design task."""
        task_dir = data_dir / self.task_id
        task_dir.mkdir(parents=True, exist_ok=True)
        
        # Create a README file with task instructions
        with open(task_dir / "README.md", "w") as f:
            f.write(f"# {self.name}\n\n")
            f.write(f"{self.description}\n\n")
            f.write(f"## Field: {self.field}\n\n")
            f.write("## Expected Output\n\n")
            f.write(f"{self.expected_output}\n\n")
        
        # Create a template for the experimental design
        template_dir = task_dir / "templates"
        template_dir.mkdir(exist_ok=True)
        
        with open(template_dir / "experimental_design_template.md", "w") as f:
            f.write(f"# Experimental Design: {self.name}\n\n")
            f.write("## Research Question\n\n[State the research question(s) to be addressed]\n\n")
            f.write("## Hypotheses\n\n[State the null and alternative hypotheses]\n\n")
            f.write("## Variables\n\n[Identify and define independent, dependent, and control variables]\n\n")
            f.write("## Experimental Setup\n\n[Describe the experimental setup and conditions]\n\n")
            f.write("## Sampling Strategy\n\n[Describe the sampling method and sample size determination]\n\n")
            f.write("## Data Collection\n\n[Describe how data will be collected and measured]\n\n")
            f.write("## Analysis Plan\n\n[Describe the statistical methods to be used for data analysis]\n\n")
            f.write("## Ethical Considerations\n\n[Address ethical issues and how they will be handled]\n\n")
            f.write("## Timeline and Resources\n\n[Outline the timeline and resources needed for the experiment]\n\n")


class NLPTask(ResearchTask):
    """Base class for natural language processing tasks."""

    def __init__(self, task_id: str, name: str, description: str, nlp_type: str, dataset_ids: Optional[List[str]] = None):
        super().__init__(task_id, name, description, "natural-language-processing")
        self.nlp_type = nlp_type
        self.data_requirements = dataset_ids or []
        self.expected_output = "A trained NLP model with evaluation metrics and analysis report."

    def prepare(self, data_dir: Path) -> None:
        """Prepare the NLP task."""
        task_dir = data_dir / self.task_id
        task_dir.mkdir(parents=True, exist_ok=True)
        
        # Create a README file with task instructions
        with open(task_dir / "README.md", "w") as f:
            f.write(f"# {self.name}\n\n")
            f.write(f"{self.description}\n\n")
            f.write(f"## NLP Type: {self.nlp_type}\n\n")
            if self.data_requirements:
                f.write("## Datasets\n\n")
                for dataset_id in self.data_requirements:
                    f.write(f"- {dataset_id}\n")
            f.write("\n## Expected Output\n\n")
            f.write(f"{self.expected_output}\n\n")
        
        # Create a template for the NLP report
        template_dir = task_dir / "templates"
        template_dir.mkdir(exist_ok=True)
        
        with open(template_dir / "nlp_report_template.md", "w") as f:
            f.write(f"# NLP Task Report: {self.name}\n\n")
            f.write("## Introduction\n\n[Provide an introduction to the NLP task and its objectives]\n\n")
            f.write("## Data Description\n\n[Describe the datasets used for training and evaluation]\n\n")
            f.write("## Methodology\n\n[Describe the NLP approach, models, and techniques used]\n\n")
            f.write("## Data Preprocessing\n\n[Describe how the text data was preprocessed]\n\n")
            f.write("## Model Architecture\n\n[Describe the architecture of the NLP model(s)]\n\n")
            f.write("## Training Process\n\n[Describe how the model was trained, including hyperparameters]\n\n")
            f.write("## Evaluation\n\n[Present evaluation metrics and results]\n\n")
            f.write("## Error Analysis\n\n[Analyze model errors and limitations]\n\n")
            f.write("## Discussion\n\n[Interpret the results and discuss their implications]\n\n")
            f.write("## Conclusion\n\n[Summarize the main findings and their significance]\n\n")