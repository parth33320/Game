import pytest
import asyncio
import os
import json
import time

from emulator.ram_scraper import RAMScraper
from ai.llm_player import LLMPlayer, VALID_ACTIONS
from chat.chat_listener import RestreamChatListener
from audit.audit_logger import AuditLogger
from scripts.run_ai_gameplay import GameplayAutomationLoop
from scripts.stream_gameplay import FFmpegRestreamStreamer

def test_ram_scraper_read_and_speed():
    memory = {
        "0x0200": 1,  # text box open
        "0x0202": ["Fight", "Run"],
        "0x0210": 15, # x
        "0x0212": 30, # y
        "0x0220": 1,  # is_battle
        "0x0222": 45, # hp
        "0x0224": 50, # max hp
        "0x0226": 12  # level
    }
    scraper = RAMScraper(memory_backend=memory)
    state = scraper.read_ram_state()

    assert state["text_box_open"] is True
    assert state["is_battle"] is True
    assert state["active_menu_options"] == ["Fight", "Run"]
    assert state["player_coords"] == {"x": 15, "y": 30}
    assert state["opponent_stats"] == {"hp": 45, "max_hp": 50, "level": 12}

    # Test speed toggle
    assert scraper.get_speed() == 1.0
    scraper.set_speed(8.0)
    assert scraper.get_speed() == 8.0

def test_llm_player_prompt_and_parser():
    player = LLMPlayer(persona="comedic hero")
    ram_state = {
        "text_box_open": True,
        "is_battle": True,
        "active_menu_options": ["Attack", "Item"],
        "player_coords": {"x": 5, "y": 5},
        "opponent_stats": {"hp": 100, "max_hp": 100, "level": 20}
    }

    prompt = player.build_prompt(ram_state)
    assert "CURRENT RAM STATE" in prompt
    assert "comedic hero" in prompt

    # Test parse valid JSON
    valid_json_resp = '{"action": "START", "dialogue": "Let us begin!"}'
    parsed = player.parse_response(valid_json_resp)
    assert parsed["action"] == "START"
    assert parsed["dialogue"] == "Let us begin!"

    # Test regex fallback
    raw_text_resp = "I think we should press B right now!"
    parsed_fallback = player.parse_response(raw_text_resp)
    assert parsed_fallback["action"] == "B"

def test_restream_chat_listener_parsing_and_queue():
    listener = RestreamChatListener(mode="anarchy")

    # Ingest invalid
    assert not listener.ingest_payload({"text": "Hello world!"})

    # Ingest valid Twitch/Restream messages
    msg1 = {"text": "press A please!", "author": "User1", "platform": "twitch"}
    msg2 = {"text": "go LEFT now", "author": "User2", "platform": "youtube"}

    assert listener.ingest_payload(msg1)
    assert listener.ingest_payload(msg2)
    assert listener.has_pending_commands()

    cmd1 = listener.get_next_command()
    assert cmd1["command"] == "A"
    assert cmd1["author"] == "User1"
    assert cmd1["platform"] == "twitch"

    cmd2 = listener.get_next_command()
    assert cmd2["command"] == "LEFT"

    assert not listener.has_pending_commands()

def test_restream_chat_democracy_voting():
    listener = RestreamChatListener(mode="democracy")

    listener.ingest_payload({"text": "UP", "author": "User1", "platform": "kick"})
    listener.ingest_payload({"text": "UP", "author": "User2", "platform": "twitch"})
    listener.ingest_payload({"text": "DOWN", "author": "User3", "platform": "youtube"})

    cmd = listener.get_next_command()
    assert cmd["command"] == "UP"
    assert cmd["votes"] == 2
    assert cmd["total_votes"] == 3

def test_chat_override_priority_in_gameplay_loop(tmp_path):
    log_file = os.path.join(tmp_path, "audit.jsonl")
    audit = AuditLogger(log_filepath=log_file)

    memory = {"0x0200": 1}  # Text box open, requires decision
    scraper = RAMScraper(memory_backend=memory)
    llm = LLMPlayer()
    chat = RestreamChatListener()

    # Add chat command
    chat.ingest_payload({"text": "SELECT", "author": "SuperViewer", "platform": "restream"})

    loop = GameplayAutomationLoop(
        ram_scraper=scraper,
        llm_player=llm,
        chat_listener=chat,
        audit_logger=audit
    )

    record = asyncio.run(loop.step())

    # Chat override MUST preempt local LLM
    assert record["source"] == "chat_override"
    assert record["action"] == "SELECT"
    assert "SuperViewer" in record["dialogue"]

def test_ffmpeg_restream_command():
    streamer = FFmpegRestreamStreamer(
        stream_key="my_key",
        rtmp_url="rtmp://live.restream.io/live"
    )
    cmd = streamer.build_ffmpeg_command()

    assert cmd[0] == "ffmpeg"
    assert "rtmp://live.restream.io/live/my_key" in cmd
    assert "libx264" in cmd
    assert "aac" in cmd
