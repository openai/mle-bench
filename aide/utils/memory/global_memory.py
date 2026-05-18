"""
Global memory layer for the search process.

Stores and retrieves node-level experience (plan, code summary, stage, label)
across the whole run for similarity search and guidance. Task-scoped (one directory per task).
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime

from .record import MemRecord
from .retriever import HybridRetriever
from .embedding_models import EmbeddingModel

logger = logging.getLogger("aide")


class GlobalMemoryLayer:
    """
    Global memory for the search: save nodes, retrieve similar/dissimilar
    records, generate guidance text. Uses MemRecord and a separate metadata map.
    """

    def __init__(
        self,
        memory_dir: str,
        embedding_model_path: str = "",
        embedding_device: str = "cuda",
        similarity_threshold: float = 0.7,
    ):
        self.memory_dir = Path(memory_dir)
        self.memory_dir.mkdir(parents=True, exist_ok=True)
        self.similarity_threshold = similarity_threshold

        self.embedding_model = EmbeddingModel(
            model_type="local",
            model_name=embedding_model_path,
            device=embedding_device,
        )
        self.retriever = HybridRetriever(self.embedding_model)

        self.records: List[MemRecord] = []
        self.node_metadata_map: Dict[str, Dict[str, Any]] = {}
        self._load_memory()

        logger.info(f"[GlobalMemory] Initialized with {len(self.records)} existing records")

    def save_node(self, node, parent_node: Optional = None) -> bool:
        """Save a search node to global memory. Returns True if saved."""
        if not self._should_save_node(node):
            return False

        try:
            code_summary = self._extract_code_summary(node)
            label = self._determine_label(node, parent_node)

            current_metric = node.metric.value if node.metric and node.metric.value is not None else None
            parent_metric = None
            if parent_node and parent_node.metric and parent_node.metric.value is not None:
                parent_metric = parent_node.metric.value

            exec_time = getattr(node, "exec_time", None)
            timestamp = datetime.now().isoformat()

            record = MemRecord(
                record_id=f"node_{node.id}",
                title=f"{node.stage} - {node.id[:8]}",
                description=node.plan or "",
                method=code_summary,
                label=label,
                timestamp=timestamp,
            )

            metadata = {
                "exec_time": exec_time,
                "parent_metric": parent_metric,
                "current_metric": current_metric,
            }
            if node.stage == "debug" and parent_node:
                parent_error = getattr(parent_node, "term_out", "") or getattr(
                    parent_node, "execution_output", ""
                )
                if parent_error:
                    metadata["parent_error"] = parent_error

            self.node_metadata_map[record.record_id] = metadata

            if record.record_id in {r.record_id for r in self.records}:
                logger.warning(f"Record {record.record_id} already exists, skipping")
                return False

            self.records.append(record)
            self._update_index(record)
            self._save_memory()

            logger.info(
                f"[GlobalMemory] Saved node {node.id} (stage={node.stage}, label={label}, "
                f"parent_metric={parent_metric}, current_metric={current_metric}, exec_time={exec_time})"
            )
            return True

        except Exception as e:
            logger.error(f"[GlobalMemory] Failed to save node {node.id}: {e}")
            return False

    def retrieve_similar_records(
        self,
        query_text: str,
        top_k: int = 2,
        alpha: float = 0.5,
        dissimilar: bool = False,
        label_filter: Optional[int] = None,
        stage_filter: Optional[str] = None,
        min_score: float = 0.0,
    ) -> List[Tuple[MemRecord, float]]:
        """Retrieve similar or dissimilar records. Returns list of (record, score)."""
        if not self.records:
            logger.info("[GlobalMemory] No records available for retrieval")
            return []

        if label_filter is not None or stage_filter is not None:
            debug_records = [
                r
                for r in self.records
                if (label_filter is None or r.label == label_filter)
                and (stage_filter is None or self._extract_stage_from_record(r) == stage_filter)
            ]
            logger.debug(
                f"[GlobalMemory] Total records: {len(self.records)}, matching "
                f"label={label_filter}, stage={stage_filter}: {len(debug_records)} records"
            )

        all_results = self.retriever.search(query_text, top_k=len(self.records), alpha=alpha)
        logger.debug(f"[GlobalMemory] Retriever returned {len(all_results)} results for query (length={len(query_text)})")

        if label_filter is not None:
            before_label = len(all_results)
            all_results = [(r, s) for r, s in all_results if r.label == label_filter]
            logger.debug(f"[GlobalMemory] After label_filter={label_filter}: {len(all_results)}/{before_label} records")
        if stage_filter is not None:
            before_stage = len(all_results)
            all_results = [
                (r, s) for r, s in all_results
                if self._extract_stage_from_record(r) == stage_filter
            ]
            logger.debug(f"[GlobalMemory] After stage_filter={stage_filter}: {len(all_results)}/{before_stage} records")

        if dissimilar:
            filtered_results = list(all_results)
            filtered_results.sort(key=lambda x: x[1])
        else:
            filtered_results = [(record, score) for record, score in all_results if score >= min_score]

        result = filtered_results[:top_k]
        logger.info(
            f"[GlobalMemory] Retrieved {len(result)} records "
            f"(dissimilar={dissimilar}, label_filter={label_filter}, stage_filter={stage_filter}, min_score={min_score}, "
            f"after_label_stage_filter={len(all_results)}, after_min_score_filter={len(filtered_results)})"
        )
        return result

    def generate_guidance_prompt(
        self,
        query_text: str,
        top_k: int = 2,
        alpha: float = 0.5,
        dissimilar: bool = False,
        stage_filter: Optional[str] = None,
    ) -> str:
        """Build guidance text from retrieved records."""
        similar_records = self.retrieve_similar_records(
            query_text=query_text,
            top_k=top_k,
            alpha=alpha,
            dissimilar=dissimilar,
            stage_filter=stage_filter,
        )

        if not similar_records:
            if dissimilar:
                return "No previous records found in memory. You can explore freely."
            return "No similar records found in memory. This is a novel direction."

        if dissimilar:
            guidance_parts = [
                "## Historical Attempts:",
                "",
                "The following approaches have been tried:",
                "",
            ]
        else:
            guidance_parts = [
                "## Historical Attempts: Similar Experiences",
                "",
                "The following similar approaches have been tried:",
                "",
            ]

        for idx, (record, score) in enumerate(similar_records, 1):
            guidance_parts.append(f"**Record #{idx}**:")
            stage = self._extract_stage_from_record(record)
            guidance_parts.append(f"- Stage: {stage}")
            guidance_parts.append(f"- Plan: {record.description}...")

            if stage == "debug":
                guidance_parts.append(f"- Result: ✅ Fixed successfully")
            elif stage in ["draft", "fusion_draft"]:
                guidance_parts.append(f"- Result: ✅ Generated successfully")
            elif stage in ["improve", "evolution", "fusion"]:
                if record.label == 1:
                    result_text = "✅ Result Improved"
                elif record.label == -1:
                    result_text = "❌ Result Worsened"
                else:
                    result_text = "⚪ Result No change"
                if record.record_id in self.node_metadata_map:
                    meta = self.node_metadata_map[record.record_id]
                    pm, cm = meta.get("parent_metric"), meta.get("current_metric")
                    if pm is not None and cm is not None:
                        result_text += f" (Metric: {pm:.6f} → {cm:.6f})"
                guidance_parts.append(f"- Result: {result_text}")
            else:
                if record.label == 1:
                    result_text = "✅ Success"
                elif record.label == -1:
                    result_text = "❌ Failed"
                else:
                    result_text = "⚪ Neutral"
                if record.record_id in self.node_metadata_map:
                    cm = self.node_metadata_map[record.record_id].get("current_metric")
                    if cm is not None:
                        result_text += f" (Metric: {cm:.6f})"
                guidance_parts.append(f"- Result: {result_text}")

            guidance_parts.append("")

        return "\n".join(guidance_parts)

    def _extract_stage_from_record(self, record: MemRecord) -> str:
        if " - " in record.title:
            return record.title.split(" - ")[0]
        return "unknown"

    def _should_save_node(self, node) -> bool:
        if node.is_buggy:
            return False
        if not node.metric or node.metric.value is None:
            return False
        return True

    def _extract_code_summary(self, node) -> str:
        if hasattr(node, "code_summary") and node.code_summary:
            return node.code_summary
        if node.plan:
            return node.plan[:500]
        if hasattr(node, "code") and node.code:
            import re
            functions = re.findall(r"def\s+(\w+)", node.code)
            classes = re.findall(r"class\s+(\w+)", node.code)
            parts = []
            if classes:
                parts.append(f"Classes: {', '.join(classes[:3])}")
            if functions:
                parts.append(f"Functions: {', '.join(functions[:5])}")
            return "; ".join(parts) if parts else "Code available"
        return "No summary available"

    def _determine_label(self, node, parent_node: Optional) -> int:
        if node.stage in ("draft", "fusion_draft"):
            return 1
        if node.stage == "debug":
            if parent_node and parent_node.is_buggy and not node.is_buggy:
                return 1
            return -1
        if node.stage in ("improve", "evolution", "fusion") and parent_node and parent_node.metric and node.metric:
            bm, cm = parent_node.metric.value, node.metric.value
            if node.metric.maximize:
                if cm > bm:
                    return 1
                if cm < bm:
                    return -1
            else:
                if cm < bm:
                    return 1
                if cm > bm:
                    return -1
        return 0

    def _update_index(self, record: MemRecord) -> None:
        text = self._extract_text(record)
        if self.retriever.vector_index is not None and len(self.records) > 1:
            self.retriever.add_to_index([record], [text])
        else:
            texts = [self._extract_text(r) for r in self.records]
            self.retriever.build_index(self.records, texts)

    def _extract_text(self, record: MemRecord) -> str:
        stage = self._extract_stage_from_record(record)
        if stage == "debug":
            if record.record_id in self.node_metadata_map:
                metadata = self.node_metadata_map[record.record_id]
                parent_error = metadata.get("parent_error", "")
                if parent_error:
                    logger.debug(f"[GlobalMemory] Using parent_error for debug record {record.record_id[:8]}, error_length={len(parent_error)}")
                    return parent_error
                logger.warning(f"[GlobalMemory] Debug record {record.record_id[:8]} has no parent_error, falling back to description+method")
            else:
                logger.warning(f"[GlobalMemory] Debug record {record.record_id[:8]} not found in node_metadata_map, falling back to description+method")
        return f"{record.description}\n{record.method}"

    def _load_memory(self) -> None:
        records_file = self.memory_dir / "records.json"
        if not records_file.exists():
            logger.info("[GlobalMemory] No existing memory file found, starting fresh")
            return
        try:
            with open(records_file, "r", encoding="utf-8") as f:
                records_data = json.load(f)

            self.records = []
            self.node_metadata_map = {}

            for item in records_data:
                metadata = {}
                for key in ("exec_time", "parent_metric", "current_metric", "parent_error"):
                    if key in item:
                        metadata[key] = item.pop(key)

                record = MemRecord.from_dict(item)
                self.records.append(record)
                if metadata:
                    self.node_metadata_map[record.record_id] = metadata

            if self.records:
                texts = [self._extract_text(r) for r in self.records]
                self.retriever.build_index(self.records, texts)
                logger.info(f"[GlobalMemory] Loaded {len(self.records)} records from disk")
        except Exception as e:
            logger.error(f"[GlobalMemory] Failed to load memory: {e}")
            self.records = []

    def _save_memory(self) -> None:
        records_file = self.memory_dir / "records.json"
        try:
            records_data = []
            for r in self.records:
                d = r.to_dict()
                if r.record_id in self.node_metadata_map:
                    meta = self.node_metadata_map[r.record_id]
                    for key in ("exec_time", "parent_metric", "current_metric", "parent_error"):
                        if meta.get(key) is not None:
                            d[key] = meta[key]
                records_data.append(d)

            with open(records_file, "w", encoding="utf-8") as f:
                json.dump(records_data, f, indent=2, ensure_ascii=False)
            logger.debug(f"[GlobalMemory] Saved {len(self.records)} records to disk")
        except Exception as e:
            logger.error(f"[GlobalMemory] Failed to save memory: {e}")
