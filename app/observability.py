import os
import sys
import json
import time
import uuid
import logging
from typing import Dict, Any, Optional

# Structured JSON Logger
class JSONFormatter(logging.Formatter):
    """Formats log records as structured JSON."""
    def format(self, record: logging.LogRecord) -> str:
        log_obj = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(record.created)),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage()
        }
        if hasattr(record, "request_id"):
            log_obj["request_id"] = record.request_id
        if hasattr(record, "extra_data"):
            log_obj.update(record.extra_data)
        if record.exc_info:
            log_obj["exception"] = self.formatException(record.exc_info)
        return json.dumps(log_obj, ensure_ascii=False)


logger = logging.getLogger("rag_production")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler(sys.stdout)
handler.setFormatter(JSONFormatter())
if not logger.handlers:
    logger.addHandler(handler)


class TelemetryTracker:
    """In-memory telemetry counters and metrics."""

    def __init__(self):
        self.start_time = time.time()
        self.total_queries = 0
        self.cache_hits = 0
        self.refusals = 0
        self.feedback_up = 0
        self.feedback_down = 0
        self.latencies = []

    def record_query(self, latency_ms: float, cached: bool = False, is_refusal: bool = False):
        self.total_queries += 1
        if cached:
            self.cache_hits += 1
        if is_refusal:
            self.refusals += 1
        self.latencies.append(latency_ms)
        if len(self.latencies) > 1000:
            self.latencies = self.latencies[-1000:]

    def record_feedback(self, rating: int):
        if rating > 0:
            self.feedback_up += 1
        elif rating < 0:
            self.feedback_down += 1

    def get_metrics(self) -> Dict[str, Any]:
        uptime = time.time() - self.start_time
        avg_lat = sum(self.latencies) / max(1, len(self.latencies)) if self.latencies else 0.0
        return {
            "uptime_seconds": round(uptime, 2),
            "total_queries": self.total_queries,
            "cache_hits": self.cache_hits,
            "cache_hit_ratio": round(self.cache_hits / max(1, self.total_queries), 4),
            "refusals": self.refusals,
            "refusal_rate": round(self.refusals / max(1, self.total_queries), 4),
            "feedback_positive": self.feedback_up,
            "feedback_negative": self.feedback_down,
            "average_latency_ms": round(avg_lat, 2)
        }


telemetry = TelemetryTracker()
