import asyncio
import time
import os
from playwright.async_api import async_playwright
from telemetry.telemetry_overlay import TelemetryPublisher

async def run_demo():
    # 1. Start Telemetry Publisher Server
    publisher = TelemetryPublisher(host="127.0.0.1", port=8085)
    publisher.start_server()
    print("Telemetry Publisher server started on http://127.0.0.1:8085")

    # Update initial telemetry data
    publisher.update_telemetry(
        ram_stats={"hp": 16, "max_hp": 16, "score": 1250, "player_coords": {"x": 340, "y": 120}},
        ai_status={"last_action": "ATTACK", "last_decision_source": "PPO_Policy", "last_dialogue": "Slashing Belmont Whip!"},
        training_status={"epoch": 100, "loss": 0.024, "curriculum_stage": 2, "best_x_pos": 1850, "retraining_active": False},
        recent_log_entry={"source": "PPO", "action": "RIGHT", "dialogue": "Advancing through stage 2..."}
    )

    screenshot_path = "telemetry_demo_screenshot.png"

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context(viewport={"width": 1280, "height": 720})
        page = await context.new_page()

        print("Navigating to Telemetry Web Dashboard Overlay...")
        await page.goto("http://127.0.0.1:8085/")
        await page.wait_for_timeout(1000)

        # Verify page elements
        title = await page.inner_text("#dashboard-title")
        print("Dashboard Title:", title)

        # Click Trigger Retraining button
        print("Clicking 'TRIGGER RL RETRAINING LOOP' button in Web UI...")
        await page.click("#btn-retrain")
        await page.wait_for_timeout(1500)

        # Take screenshot of the interactive web UI dashboard
        await page.screenshot(path=screenshot_path, full_page=True)
        print(f"Captured Playwright screenshot at {screenshot_path}")

        await browser.close()

    publisher.stop_server()
    print("Telemetry Publisher server stopped.")

if __name__ == "__main__":
    asyncio.run(run_demo())
