import pytest
import time
import subprocess
import sys
import os

from scripts.watchdog import ProcessWatchdog
from audit.audit_logger import AuditLogger

def test_watchdog_subprocess_crash_recovery_and_audit(tmp_path):
    log_file = str(tmp_path / "watchdog_audit.jsonl")
    audit = AuditLogger(log_filepath=log_file)

    cmd = [sys.executable, "-c", "import sys; sys.exit(1)"]
    watchdog = ProcessWatchdog(
        managed_services={"short_lived_service": cmd},
        check_interval=0.1,
        audit_logger=audit
    )

    watchdog.start_all()
    time.sleep(0.3)
    
    status1 = watchdog.check_health_and_recover()
    assert "short_lived_service" in status1
    assert status1["short_lived_service"]["restarts"] >= 1

    watchdog.stop_all()

    events = audit.read_all_events()
    event_types = [e.get("event_type") or e.get("details", {}).get("event") for e in events]
    assert "service_started" in event_types or "service_crashed" in event_types or "watchdog_recovery" in event_types

def test_dockerfile_and_compose_structure():
    assert os.path.exists("docker/Dockerfile")
    assert os.path.exists("docker-compose.yml")

    with open("docker/Dockerfile", "r") as f:
        df_content = f.read()
        assert "FROM python:3.10-slim AS builder" in df_content
        assert "FROM python:3.10-slim AS final" in df_content
        assert "HEALTHCHECK" in df_content

    with open("docker-compose.yml", "r") as f:
        dc_content = f.read()
        assert "services:" in dc_content
        assert "app:" in dc_content
        assert "healthcheck:" in dc_content
