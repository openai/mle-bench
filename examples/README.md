# Examples

## Rule Violation Detection

Given a run group produced by `python run_agent.py`, you can use `examples/rule_violation_detection/analyze_logs.sh` to detect rule violations in the agent's logs. Adjust the `mlebench_dir` and `log_dir` variables to point to the appropriate directories. This script first calls `examples/rule_violation_detection/make_logs_file.py` to produce a single `logs.jsonl` file by concatenating agent logs from each competition, then runs the rule violation detector on the resulting `logs.jsonl` file.
