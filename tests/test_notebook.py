import json
import os
import pytest

def test_colab_notebook_structure():
    notebook_path = os.path.join("notebooks", "castlevania_colab_trainer.ipynb")
    assert os.path.exists(notebook_path), f"{notebook_path} does not exist!"

    with open(notebook_path, "r", encoding="utf-8") as f:
        nb = json.load(f)

    assert "cells" in nb
    cells = nb["cells"]
    assert len(cells) == 12, f"Expected 12 cells (6 markdown + 6 code), got {len(cells)}"

    # Check cell types alternating markdown and code
    for i in range(6):
        md_cell = cells[i * 2]
        code_cell = cells[i * 2 + 1]

        assert md_cell["cell_type"] == "markdown", f"Cell {i*2+1} should be markdown"
        assert code_cell["cell_type"] == "code", f"Cell {i*2+2} should be code"

        md_text = "".join(md_cell["source"])
        code_text = "".join(code_cell["source"])

        assert f"Cell {i+1}:" in md_text, f"Cell {i*2+1} markdown missing 'Cell {i+1}:' header"

    # Specific keyword checks per cell
    c1_code = "".join(cells[1]["source"])
    assert "torch.cuda.is_available()" in c1_code
    assert "torch.cuda.get_device_name(0)" in c1_code
    assert "apt-get install -y ffmpeg python3-opengl xvfb libsdl2-dev" in c1_code
    assert "pip install stable-retro gymnasium torch matplotlib" in c1_code

    c2_code = "".join(cells[3]["source"])
    assert "pyvirtualdisplay" in c2_code or "Display" in c2_code

    c3_code = "".join(cells[5]["source"])
    assert "Castlevania (USA).nes" in c3_code
    assert "retro.data.merge_into_master_python_where_necessary" in c3_code

    c4_code = "".join(cells[7]["source"])
    assert "google.colab" in c4_code
    assert "drive.mount" in c4_code
    assert "/content/drive/MyDrive/Castlevania_RL_Checkpoints" in c4_code

    c5_code = "".join(cells[9]["source"])
    assert "ppo_agent_ep5000.pt" in c5_code
    assert "best_ppo_agent_dist" in c5_code

    c6_code = "".join(cells[11]["source"])
    assert "xvfb-run" in c6_code
    assert "scripts/watchdog.py" in c6_code
    assert "--use-retro" in c6_code
    assert "--cuda" in c6_code
