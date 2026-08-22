import json
import time
import threading
from http.server import HTTPServer, BaseHTTPRequestHandler
from typing import Dict, Any, List, Optional

class TelemetryPublisher:
    """
    Real-time telemetry state publisher that aggregates emulator RAM stats,
    recent AI/LLM decision log, viewer override status, system operational metrics,
    and active RL training / retraining loop metrics.
    Exposes data via JSON feed, supports control endpoints, and serves an interactive web overlay dashboard.
    """
    def __init__(self, host: str = "0.0.0.0", port: int = 8080):
        self.host = host
        self.port = port
        self._lock = threading.Lock()
        self.current_state: Dict[str, Any] = {
            "timestamp": time.time(),
            "status": "OPERATIONAL",
            "ram_stats": {
                "hp": 100,
                "max_hp": 100,
                "level": 1,
                "score": 0,
                "player_coords": {"x": 0, "y": 0},
                "active_threats": []
            },
            "ai_status": {
                "active_persona": "comedic hero",
                "last_decision_source": "system_init",
                "last_action": "NONE",
                "last_dialogue": "Initializing telemetry pipeline..."
            },
            "training_status": {
                "epoch": 0,
                "loss": 0.0,
                "curriculum_stage": 1,
                "best_x_pos": 0,
                "retraining_active": False,
                "last_retrain_time": None,
                "retrain_reason": None
            },
            "recent_decision_log": [],
            "override_status": {
                "active_override": False,
                "override_type": None,
                "winner": None,
                "remaining_seconds": 0.0
            },
            "system_metrics": {
                "speed_mode": 1.0,
                "fps": 60,
                "uptime_seconds": 0.0
            }
        }
        self.start_time = time.time()
        self.server: Optional[HTTPServer] = None
        self.server_thread: Optional[threading.Thread] = None

    def update_telemetry(
        self,
        ram_stats: Optional[Dict[str, Any]] = None,
        ai_status: Optional[Dict[str, Any]] = None,
        training_status: Optional[Dict[str, Any]] = None,
        recent_log_entry: Optional[Dict[str, Any]] = None,
        override_status: Optional[Dict[str, Any]] = None,
        speed_mode: Optional[float] = None,
        status: Optional[str] = None
    ):
        """Thread-safe update of current telemetry state."""
        with self._lock:
            self.current_state["timestamp"] = time.time()
            self.current_state["system_metrics"]["uptime_seconds"] = round(time.time() - self.start_time, 2)
            
            if status is not None:
                self.current_state["status"] = status

            if speed_mode is not None:
                self.current_state["system_metrics"]["speed_mode"] = speed_mode

            if ram_stats is not None:
                self.current_state["ram_stats"].update(ram_stats)

            if ai_status is not None:
                self.current_state["ai_status"].update(ai_status)

            if training_status is not None:
                self.current_state["training_status"].update(training_status)

            if override_status is not None:
                self.current_state["override_status"].update(override_status)

            if recent_log_entry is not None:
                log = self.current_state["recent_decision_log"]
                log.insert(0, recent_log_entry)
                self.current_state["recent_decision_log"] = log[:10]

    def trigger_retrain(self, reason: str = "Manual Trigger"):
        """Triggers retraining loop in training_status."""
        with self._lock:
            self.current_state["training_status"]["retraining_active"] = True
            self.current_state["training_status"]["last_retrain_time"] = time.time()
            self.current_state["training_status"]["retrain_reason"] = reason

    def get_state_snapshot(self) -> Dict[str, Any]:
        """Returns a copy of current telemetry state."""
        with self._lock:
            return json.loads(json.dumps(self.current_state))

    def _make_handler(publisher_self):
        class TelemetryHTTPRequestHandler(BaseHTTPRequestHandler):
            def log_message(self, format, *args):
                pass

            def do_GET(self):
                if self.path in ("/api/telemetry", "/telemetry.json"):
                    data = publisher_self.get_state_snapshot()
                    body = json.dumps(data).encode("utf-8")
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Access-Control-Allow-Origin", "*")
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                elif self.path in ("/overlay", "/"):
                    html = publisher_self.get_html_overlay()
                    body = html.encode("utf-8")
                    self.send_response(200)
                    self.send_header("Content-Type", "text/html")
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                else:
                    self.send_response(404)
                    self.end_headers()

            def do_POST(self):
                content_length = int(self.headers.get("Content-Length", 0))
                raw_body = self.rfile.read(content_length) if content_length > 0 else b"{}"
                try:
                    payload = json.loads(raw_body.decode("utf-8"))
                except Exception:
                    payload = {}

                if self.path == "/api/trigger_retrain":
                    reason = payload.get("reason", "Manual Web Dashboard Trigger")
                    publisher_self.trigger_retrain(reason=reason)
                    resp = json.dumps({"status": "retraining_triggered", "reason": reason}).encode("utf-8")
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Access-Control-Allow-Origin", "*")
                    self.send_header("Content-Length", str(len(resp)))
                    self.end_headers()
                    self.wfile.write(resp)

                elif self.path == "/api/update_training":
                    publisher_self.update_telemetry(training_status=payload)
                    resp = json.dumps({"status": "updated"}).encode("utf-8")
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Access-Control-Allow-Origin", "*")
                    self.send_header("Content-Length", str(len(resp)))
                    self.end_headers()
                    self.wfile.write(resp)

                else:
                    self.send_response(404)
                    self.end_headers()

        return TelemetryHTTPRequestHandler

    def get_html_overlay(self) -> str:
        """Generates an interactive web HTML overlay & telemetry dashboard UI."""
        return """<!DOCTYPE html>
<html>
<head>
    <title>Stream Telemetry & RL Control Dashboard</title>
    <style>
        body {
            font-family: 'Courier New', Courier, monospace;
            background-color: #0d1117;
            color: #00FF66;
            margin: 15px;
            padding: 15px;
            border-radius: 8px;
            max-width: 500px;
            box-shadow: 0 0 15px rgba(0, 255, 102, 0.4);
        }
        h2 { margin-top: 0; color: #00FFFF; text-align: center; border-bottom: 2px solid #00FFFF; padding-bottom: 5px; }
        .section { margin-bottom: 12px; background: #161b22; padding: 10px; border-radius: 6px; border: 1px solid #30363d; }
        .label { font-weight: bold; color: #FFCC00; }
        .value { color: #FFFFFF; }
        .override { color: #FF3366; font-weight: bold; }
        .retrain-banner { color: #FF9900; font-weight: bold; }
        .log-list { font-size: 11px; max-height: 100px; overflow-y: hidden; }
        .log-item { margin-bottom: 4px; border-bottom: 1px dashed #333; }
        button {
            background-color: #238636;
            color: #ffffff;
            border: none;
            padding: 8px 16px;
            font-family: inherit;
            font-size: 12px;
            font-weight: bold;
            border-radius: 4px;
            cursor: pointer;
            width: 100%;
            margin-top: 5px;
        }
        button:hover {
            background-color: #2ea043;
        }
    </style>
</head>
<body>
    <h2 id="dashboard-title">24/7 AI TELEMETRY & RL CONTROL</h2>
    <div class="section">
        <span class="label">STATUS:</span> <span id="status" class="value">--</span> |
        <span class="label">SPEED:</span> <span id="speed" class="value">--</span> |
        <span class="label">UPTIME:</span> <span id="uptime" class="value">--</span>s
    </div>
    <div class="section">
        <span class="label">HP:</span> <span id="hp" class="value">--</span> |
        <span class="label">SCORE:</span> <span id="score" class="value">--</span><br>
        <span class="label">POS:</span> <span id="coords" class="value">--</span> |
        <span class="label">THREATS:</span> <span id="threats" class="value">--</span>
    </div>
    <div class="section">
        <span class="label">RL TRAINING & RETRAINING LOOP:</span><br>
        <span class="label">EPOCH:</span> <span id="training-epoch" class="value">--</span> |
        <span class="label">LOSS:</span> <span id="training-loss" class="value">--</span><br>
        <span class="label">STAGE:</span> <span id="curriculum-stage" class="value">--</span> |
        <span class="label">MAX X POS:</span> <span id="best-x-pos" class="value">--</span><br>
        <span id="retrain-status" class="value">--</span>
        <button id="btn-retrain" onclick="triggerRetrain()">TRIGGER RL RETRAINING LOOP</button>
    </div>
    <div class="section">
        <span class="label">OVERRIDE STATUS:</span> <span id="override" class="value">--</span>
    </div>
    <div class="section">
        <span class="label">LAST ACTION:</span> <span id="action" class="value">--</span><br>
        <span class="label">DIALOGUE:</span> <span id="dialogue" class="value">--</span>
    </div>
    <div class="section">
        <span class="label">RECENT DECISION LOG:</span>
        <div id="log" class="log-list"></div>
    </div>

    <script>
        async function fetchTelemetry() {
            try {
                const res = await fetch('/api/telemetry');
                const data = await res.json();
                document.getElementById('status').innerText = data.status;
                document.getElementById('speed').innerText = data.system_metrics.speed_mode + 'x';
                document.getElementById('uptime').innerText = data.system_metrics.uptime_seconds;
                document.getElementById('hp').innerText = data.ram_stats.hp + '/' + data.ram_stats.max_hp;
                document.getElementById('score').innerText = data.ram_stats.score;
                document.getElementById('coords').innerText = `(${data.ram_stats.player_coords.x}, ${data.ram_stats.player_coords.y})`;
                document.getElementById('threats').innerText = (data.ram_stats.active_threats || []).length;

                const tr = data.training_status || {};
                document.getElementById('training-epoch').innerText = tr.epoch !== undefined ? tr.epoch : '--';
                document.getElementById('training-loss').innerText = tr.loss !== undefined ? tr.loss : '--';
                document.getElementById('curriculum-stage').innerText = tr.curriculum_stage !== undefined ? tr.curriculum_stage : '--';
                document.getElementById('best-x-pos').innerText = tr.best_x_pos !== undefined ? tr.best_x_pos : '--';

                if (tr.retraining_active) {
                    document.getElementById('retrain-status').innerHTML = `<span class="retrain-banner">[RETRAINING LOOP IN PROGRESS: ${tr.retrain_reason || 'Active'}]</span>`;
                } else {
                    document.getElementById('retrain-status').innerText = "RL Model Status: Operational / Idle";
                }

                const ov = data.override_status;
                if (ov.active_override) {
                    document.getElementById('override').innerHTML = `<span class="override">EXCLUSIVE CONTROL BY ${ov.winner} (${ov.remaining_seconds}s)</span>`;
                } else {
                    document.getElementById('override').innerText = "AI / ANARCHY MODE ACTIVE";
                }

                document.getElementById('action').innerText = data.ai_status.last_action + " (" + data.ai_status.last_decision_source + ")";
                document.getElementById('dialogue').innerText = data.ai_status.last_dialogue;

                const logContainer = document.getElementById('log');
                logContainer.innerHTML = (data.recent_decision_log || []).slice(0, 5).map(item =>
                    `<div class="log-item">[${item.source || 'AI'}] ${item.action || ''}: ${item.dialogue || ''}</div>`
                ).join('');
            } catch (e) {
                console.error("Telemetry fetch error:", e);
            }
        }

        async function triggerRetrain() {
            try {
                const res = await fetch('/api/trigger_retrain', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ reason: 'Dashboard Button Clicked' })
                });
                const data = await res.json();
                console.log("Retrain triggered:", data);
                fetchTelemetry();
            } catch (e) {
                console.error("Error triggering retrain:", e);
            }
        }

        setInterval(fetchTelemetry, 1000);
        fetchTelemetry();
    </script>
</body>
</html>
"""

    def start_server(self):
        handler_class = self._make_handler()
        self.server = HTTPServer((self.host, self.port), handler_class)
        self.server_thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.server_thread.start()

    def stop_server(self):
        if self.server:
            self.server.shutdown()
            self.server.server_close()
