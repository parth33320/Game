import pytest
import asyncio
import time
import json
import urllib.request
import urllib.parse

from telemetry.telemetry_overlay import TelemetryPublisher
from chat.challenge_evaluator import ChallengeEvaluator, SafeASTEvaluator
from chat.chat_listener import RestreamChatListener
from scripts.run_ai_gameplay import GameplayAutomationLoop, PPOPolicyRunner
from emulator.ram_scraper import RAMScraper
from ai.llm_player import LLMPlayer
from audit.audit_logger import AuditLogger

def test_telemetry_publisher_state_and_http_server():
    publisher = TelemetryPublisher(host="127.0.0.1", port=8089)
    publisher.update_telemetry(
        ram_stats={"hp": 75, "score": 250, "player_coords": {"x": 10, "y": 20}},
        ai_status={"last_action": "ATTACK", "last_dialogue": "Defeating monster!"},
        speed_mode=2.0
    )

    snapshot = publisher.get_state_snapshot()
    assert snapshot["ram_stats"]["hp"] == 75
    assert snapshot["ram_stats"]["score"] == 250
    assert snapshot["ai_status"]["last_action"] == "ATTACK"
    assert snapshot["system_metrics"]["speed_mode"] == 2.0

    publisher.start_server()
    time.sleep(0.2)
    try:
        req = urllib.request.urlopen("http://127.0.0.1:8089/api/telemetry")
        assert req.status == 200
        data = json.loads(req.read().decode("utf-8"))
        assert data["ram_stats"]["hp"] == 75

        req_html = urllib.request.urlopen("http://127.0.0.1:8089/overlay")
        assert req_html.status == 200
        html = req_html.read().decode("utf-8")
        assert "24/7 AI TELEMETRY" in html
    finally:
        publisher.stop_server()

def test_telemetry_retraining_loop_api_and_dashboard():
    publisher = TelemetryPublisher(host="127.0.0.1", port=8090)
    publisher.start_server()
    time.sleep(0.2)
    try:
        # Check initial training state in snapshot
        snapshot = publisher.get_state_snapshot()
        assert "training_status" in snapshot
        assert snapshot["training_status"]["retraining_active"] is False

        # POST to update training stats
        post_data = json.dumps({
            "epoch": 42,
            "loss": 0.015,
            "curriculum_stage": 3,
            "best_x_pos": 1420
        }).encode("utf-8")
        req = urllib.request.Request(
            "http://127.0.0.1:8090/api/update_training",
            data=post_data,
            headers={"Content-Type": "application/json"},
            method="POST"
        )
        with urllib.request.urlopen(req) as resp:
            assert resp.status == 200
            res_json = json.loads(resp.read().decode("utf-8"))
            assert res_json["status"] == "updated"

        # Verify snapshot updated
        snap2 = publisher.get_state_snapshot()
        assert snap2["training_status"]["epoch"] == 42
        assert snap2["training_status"]["loss"] == 0.015
        assert snap2["training_status"]["best_x_pos"] == 1420

        # POST trigger retrain
        retrain_req = urllib.request.Request(
            "http://127.0.0.1:8090/api/trigger_retrain",
            data=json.dumps({"reason": "Manual Trigger"}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST"
        )
        with urllib.request.urlopen(retrain_req) as resp:
            assert resp.status == 200
            res_json = json.loads(resp.read().decode("utf-8"))
            assert res_json["status"] == "retraining_triggered"

        snap3 = publisher.get_state_snapshot()
        assert snap3["training_status"]["retraining_active"] is True

        # Check HTML overlay UI contains retrain button and training section
        req_html = urllib.request.urlopen("http://127.0.0.1:8090/")
        html = req_html.read().decode("utf-8")
        assert "btn-retrain" in html
        assert "training-epoch" in html
    finally:
        publisher.stop_server()

def test_safe_ast_evaluator():
    success1, val1 = SafeASTEvaluator.safe_eval("[1, 2, 3]")
    assert success1 is True
    assert val1 == [1, 2, 3]

    success_evil, _ = SafeASTEvaluator.safe_eval("__import__('os').system('echo hack')")
    assert success_evil is False

def test_challenge_evaluator_grading_and_penalties():
    evaluator = ChallengeEvaluator(cooldown_seconds=5.0, reward_duration_seconds=20.0)
    
    evaluator.active_challenge = {
        "details": {
            "id": "stem_01",
            "category": "STEM Trivia",
            "prompt": "STEM Trivia: What is the atomic number of Carbon?",
            "expected": r"^6$",
            "type": "regex"
        },
        "start_time": time.time(),
        "solved": False,
        "winner": None
    }

    wrong_res = evaluator.evaluate_submission("Bob", "12")
    assert wrong_res["correct"] is False
    assert wrong_res["status"] == "incorrect"
    assert "Penalty" in wrong_res["message"]

    cooldown_res = evaluator.evaluate_submission("Bob", "6")
    assert cooldown_res["correct"] is False
    assert cooldown_res["status"] == "cooldown"

    correct_res = evaluator.evaluate_submission("Alice", "6")
    assert correct_res["correct"] is True
    assert correct_res["winner"] == "Alice"
    assert correct_res["reward_duration"] == 20.0

def test_exclusive_winner_override_precedence_and_timers(tmp_path):
    log_file = str(tmp_path / "audit.jsonl")
    audit = AuditLogger(log_filepath=log_file)
    evaluator = ChallengeEvaluator(reward_duration_seconds=2.0)
    chat = RestreamChatListener(challenge_evaluator=evaluator)

    evaluator.active_challenge = {
        "details": {
            "id": "test_01",
            "category": "Trivia",
            "prompt": "Press 1",
            "expected": r"^1$",
            "type": "regex"
        },
        "start_time": time.time(),
        "solved": False,
        "winner": None
    }

    chat.ingest_payload({"text": "!answer 1", "author": "Alice", "platform": "twitch"})
    
    override_info = chat.get_exclusive_override_info()
    assert override_info["active_override"] is True
    assert override_info["winner"] == "Alice"

    chat.ingest_payload({"text": "DOWN", "author": "Bob", "platform": "youtube"})
    chat.ingest_payload({"text": "UP", "author": "Alice", "platform": "twitch"})

    cmd = chat.get_next_command()
    assert cmd["command"] == "UP"
    assert cmd["author"] == "Alice"
    assert chat.get_next_command() is None

def test_ppo_autonomous_fallback_non_blocking_during_decisions(tmp_path):
    log_file = str(tmp_path / "audit.jsonl")
    audit = AuditLogger(log_filepath=log_file)

    memory = {"0x0200": 1, "0x0220": 1}
    scraper = RAMScraper(memory_backend=memory)
    
    loop = GameplayAutomationLoop(
        ram_scraper=scraper,
        enable_llm_fallback=False,
        audit_logger=audit
    )

    record = asyncio.run(loop.step())
    
    assert record["source"] == "ppo_policy"
    assert record["action"] == "A"
    assert "PPO Policy" in record["dialogue"]
