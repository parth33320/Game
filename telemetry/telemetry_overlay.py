import json
import time
import threading
from http.server import HTTPServer, BaseHTTPRequestHandler
from typing import Dict, Any, List, Optional

class TelemetryPublisher:
    """
    Real-time telemetry state publisher that aggregates emulator RAM stats,
    recent AI/LLM decision log, viewer override status, and system operational metrics.
    Exposes data via JSON feed and serves a lightweight HTML overlay for stream ingestion.
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

            if override_status is not None:
                self.current_state["override_status"].update(override_status)

            if recent_log_entry is not None:
                log = self.current_state["recent_decision_log"]
                log.insert(0, recent_log_entry)
                self.current_state["recent_decision_log"] = log[:10]

    def get_state_snapshot(self) -> Dict[str, Any]:
        """Returns a copy of current telemetry state."""
        with self._lock:
            return json.loads(json.dumps(self.current_state))

    def _make_handler(publisher_self):
        class TelemetryHTTPRequestHandler(BaseHTTPRequestHandler):
            def log_message(self, format, *args):
                pass

            def do_GET(self):
                if self.path == "/api/telemetry" or self.path == "/telemetry.json":
                    data = publisher_self.get_state_snapshot()
                    body = json.dumps(data).encode("utf-8")
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Access-Control-Allow-Origin", "*")
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                elif self.path == "/overlay" or self.path == "/":
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

        return TelemetryHTTPRequestHandler

    def get_html_overlay(self) -> str:
        """Generates a lightweight web HTML overlay UI for live stream display."""
        return """<!DOCTYPE html>
<html>
<head>
    <title>Stream Telemetry Overlay</title>
    <style>
        body {
            font-family: 'Courier New', Courier, monospace;
            background-color: rgba(0, 0, 0, 0.75);
            color: #00FF66;
            margin: 10px;
            padding: 10px;
            border-radius: 8px;
            width: 380px;
            box-shadow: 0 0 10px rgba(0, 255, 102, 0.4);
        }
        h2 { margin-top: 0; color: #00FFFF; text-align: center; border-bottom: 1px solid #00FFFF; }
        .section { margin-bottom: 12px; }
        .label { font-weight: bold; color: #FFCC00; }
        .value { color: #FFFFFF; }
        .override { color: #FF3366; font-weight: bold; }
        .log-list { font-size: 11px; max-height: 100px; overflow-y: hidden; }
        .log-item { margin-bottom: 4px; border-bottom: 1px dashed #333; }
    </style>
</head>
<body>
    <h2>24/7 AI TELEMETRY</h2>
    <div class="section">
        <span class="label">STATUS:</span> <span id="status" class="value">--</span> |
        <span class="label">SPEED:</span> <span id="speed" class="value">--</span>
    </div>
    <div class="section">
        <span class="label">HP:</span> <span id="hp" class="value">--</span> |
        <span class="label">SCORE:</span> <span id="score" class="value">--</span><br>
        <span class="label">POS:</span> <span id="coords" class="value">--</span> |
        <span class="label">THREATS:</span> <span id="threats" class="value">--</span>
    </div>
    <div class="section">
        <span class="label">OVERRIDE:</span> <span id="override" class="value">--</span>
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
                document.getElementById('hp').innerText = data.ram_stats.hp + '/' + data.ram_stats.max_hp;
                document.getElementById('score').innerText = data.ram_stats.score;
                document.getElementById('coords').innerText = `(${data.ram_stats.player_coords.x}, ${data.ram_stats.player_coords.y})`;
                document.getElementById('threats').innerText = (data.ram_stats.active_threats || []).length;

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
