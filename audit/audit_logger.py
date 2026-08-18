import json
import time
import os
from typing import Dict, Any

class AuditLogger:
    """
    Append-only JSONL logger tracking decision records, RAM snapshots, chat overrides,
    and system action execution for auditability.
    """
    def __init__(self, log_filepath: str = "audit_log.jsonl"):
        self.log_filepath = log_filepath
        os.makedirs(os.path.dirname(os.path.abspath(log_filepath)), exist_ok=True)

    def log_event(self, event_type: str, details: Dict[str, Any]) -> Dict[str, Any]:
        record = {
            "timestamp": time.time(),
            "event_type": event_type,
            "details": details
        }
        with open(self.log_filepath, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
        return record
