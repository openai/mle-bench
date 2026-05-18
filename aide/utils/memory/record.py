"""
Minimal memory record for search node history.

Used by GlobalMemoryLayer for storing and retrieving node-level experience
(plan, code summary, stage, label).
"""

from dataclasses import dataclass
from typing import Dict, Any, Optional


@dataclass
class MemRecord:
    """Lightweight record for a single node in memory (retrieval and display)."""
    record_id: str
    title: str       # e.g. "{stage} - {node_id[:8]}"
    description: str # plan text
    method: str      # code summary
    label: int       # 1 success, 0 neutral, -1 failure
    timestamp: Optional[str] = None

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MemRecord":
        """Build from dict (e.g. loaded from JSON). Extra keys are ignored."""
        return cls(
            record_id=data.get("record_id", ""),
            title=data.get("title", ""),
            description=data.get("description", ""),
            method=data.get("method", ""),
            label=data.get("label", 0),
            timestamp=data.get("timestamp"),
        )

    def to_dict(self) -> Dict[str, Any]:
        """For JSON serialization."""
        d = {
            "record_id": self.record_id,
            "title": self.title,
            "description": self.description,
            "method": self.method,
            "label": self.label,
        }
        if self.timestamp is not None:
            d["timestamp"] = self.timestamp
        return d
