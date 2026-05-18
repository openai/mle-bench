"""Submission validation server — validates submission files against mlebench competitions."""

import logging
import os
import tempfile
from pathlib import Path

from flask import Flask, jsonify, request
from mlebench.grade import validate_submission
from mlebench.registry import registry

from aide.utils.config import load_cfg

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

_cfg = load_cfg()
_data_dir = Path(_cfg.data_dir)


@app.post("/validate")
def validate():
    if "file" not in request.files:
        return jsonify({"error": "Missing 'file' in request"}), 400

    competition_id = request.headers.get("exp-id")
    if not competition_id:
        return jsonify({"error": "Missing 'exp-id' header"}), 400

    # Save uploaded file to a temp path (auto-cleaned)
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tmp:
        request.files["file"].save(tmp)
        tmp_path = Path(tmp.name)

    try:
        comp = registry.set_data_dir(_data_dir).get_competition(competition_id)
        is_valid, message = validate_submission(tmp_path, comp)
        logger.info(f"Validation for {competition_id}: is_valid={is_valid}, message={message}")
        return jsonify({"is_valid": is_valid, "result": message})
    except Exception as e:
        logger.exception("Validation failed")
        return jsonify({"error": "Validation failed", "details": str(e)}), 500
    finally:
        tmp_path.unlink(missing_ok=True)


@app.get("/health")
def health():
    return jsonify({"status": "running"})


if __name__ == "__main__":
    port = int(os.getenv("GRADING_SERVER_PORT", "5005"))
    app.run(host="0.0.0.0", port=port)
