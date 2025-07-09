# Research Tasks Examples

This directory contains examples of how to use the research tasks functionality in ResearchBench.

## Available Examples

- `run_literature_review.py`: Example of running a literature review research task
- `run_data_analysis.py`: Example of running a data analysis research task

## Running the Examples

### Literature Review

To run the literature review example:

```bash
python run_literature_review.py --task-id literature-review-ai-ethics --output-dir ./output
```

This will run a literature review on AI ethics and save the output to the specified directory.

### Data Analysis

To run the data analysis example:

```bash
python run_data_analysis.py --task-id data-analysis-climate-change --output-dir ./output
```

This will run a data analysis on climate change data and save the output to the specified directory.

### Social Media Sentiment Analysis

To run the social media sentiment analysis example with custom parameters:

```bash
python run_data_analysis.py --task-id data-analysis-social-media-sentiment --output-dir ./output --params '{"topic": "artificial intelligence", "time_period": "2024-01-01 to 2024-06-30", "platforms": ["Twitter", "Reddit", "LinkedIn"]}'
```

This will analyze sentiment in social media posts about artificial intelligence.

## Creating Your Own Research Tasks

You can create your own research tasks by extending the base classes in `researchbench.research_tasks.base` and registering them with the task manager using the `@register_task_handler` decorator.

See the implementation of existing tasks in the `researchbench/research_tasks/` directory for examples.