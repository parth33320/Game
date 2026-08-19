import sys
import os
import time
import subprocess
import signal
from typing import Dict, Any, Optional

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from audit.audit_logger import AuditLogger

class ProcessWatchdog:
    """
    24/7 Health Check & Auto-Recovery Supervisor Service.
    Monitors active background services (PyTorch PPO training agent loop, gameplay runner, FFmpeg streamer),
    catches segmentation faults, crashes, or unhandled exceptions, automatically restarts child processes,
    and logs all recovery incidents to the append-only audit trail.
    """
    def __init__(
        self,
        managed_services: Optional[Dict[str, list]] = None,
        check_interval: float = 2.0,
        audit_logger: Optional[AuditLogger] = None
    ):
        self.check_interval = check_interval
        self.audit_logger = audit_logger or AuditLogger()
        python_exe = sys.executable
        self.managed_services = managed_services or {
            "training_agent": [python_exe, "scripts/train_agent.py"],
            "gameplay_loop": [python_exe, "scripts/run_ai_gameplay.py"],
            "ffmpeg_streamer": [python_exe, "scripts/stream_gameplay.py"]
        }
        self.processes: Dict[str, subprocess.Popen] = {}
        self.restart_counts: Dict[str, int] = {}
        self.running = False

    def start_service(self, name: str, cmd: list) -> subprocess.Popen:
        proc = subprocess.Popen(cmd)
        self.processes[name] = proc
        self.audit_logger.log_event("service_started", {
            "service": name,
            "pid": proc.pid,
            "cmd": cmd,
            "timestamp": time.time()
        })
        return proc

    def start_all(self):
        self.running = True
        for name, cmd in self.managed_services.items():
            self.start_service(name, cmd)
            self.restart_counts[name] = 0

    def check_health_and_recover(self) -> Dict[str, Any]:
        status_report = {}
        for name, cmd in self.managed_services.items():
            proc = self.processes.get(name)
            if proc is None or proc.poll() is not None:
                exit_code = proc.poll() if proc else -1
                self.restart_counts[name] = self.restart_counts.get(name, 0) + 1
                
                incident = {
                    "event": "service_crashed",
                    "service": name,
                    "exit_code": exit_code,
                    "restart_count": self.restart_counts[name],
                    "timestamp": time.time()
                }
                self.audit_logger.log_event("watchdog_recovery", incident)
                print(f"WATCHDOG RECOVERY: Service '{name}' exited with code {exit_code}. Auto-restarting (Attempt {self.restart_counts[name]})...")
                
                new_proc = self.start_service(name, cmd)
                status_report[name] = {
                    "status": "restarted",
                    "pid": new_proc.pid,
                    "restarts": self.restart_counts[name]
                }
            else:
                status_report[name] = {
                    "status": "healthy",
                    "pid": proc.pid,
                    "restarts": self.restart_counts[name]
                }
        return status_report

    def stop_all(self):
        self.running = False
        for name, proc in self.processes.items():
            if proc and proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    proc.kill()
                self.audit_logger.log_event("service_stopped", {
                    "service": name,
                    "pid": proc.pid,
                    "timestamp": time.time()
                })

    def run_forever(self, max_cycles: Optional[int] = None):
        self.start_all()
        cycles = 0
        try:
            while self.running:
                self.check_health_and_recover()
                cycles += 1
                if max_cycles and cycles >= max_cycles:
                    break
                time.sleep(self.check_interval)
        finally:
            self.stop_all()

if __name__ == "__main__":
    watchdog = ProcessWatchdog()
    watchdog.run_forever(max_cycles=3)
