import io
import os
import zipfile
from typing import Iterable, List

ACTION_NAMES = ["NOOP", "RIGHT", "LEFT", "DOWN", "JUMP", "WHIP", "RIGHT+JUMP", "RIGHT+WHIP", "UP"]


def _action_from_buttons(buttons: str) -> int:
    pressed = set(buttons)
    right = "R" in pressed
    jump = "A" in pressed
    whip = "B" in pressed
    if right and jump:
        return ACTION_NAMES.index("RIGHT+JUMP")
    if right and whip:
        return ACTION_NAMES.index("RIGHT+WHIP")
    if right:
        return ACTION_NAMES.index("RIGHT")
    if "L" in pressed:
        return ACTION_NAMES.index("LEFT")
    if "D" in pressed:
        return ACTION_NAMES.index("DOWN")
    if "U" in pressed:
        return ACTION_NAMES.index("UP")
    if jump:
        return ACTION_NAMES.index("JUMP")
    if whip:
        return ACTION_NAMES.index("WHIP")
    return ACTION_NAMES.index("NOOP")


def parse_bk2_actions(path: str) -> List[int]:
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        input_name = next((name for name in names if name.lower().endswith("input log.txt")), None)
        if input_name is None:
            raise ValueError(f"{path} has no Input Log.txt")
        text = io.TextIOWrapper(archive.open(input_name), encoding="utf-8", errors="replace")
        actions = []
        for line in text:
            line = line.strip()
            if not line.startswith("|") or "LogKey:" in line:
                continue
            fields = line.split("|")
            if len(fields) < 3:
                continue
            actions.append(_action_from_buttons(fields[2]))
    return actions


def parse_fm2_actions(path: str) -> List[int]:
    actions = []
    with open(path, encoding="utf-8", errors="replace") as stream:
        for line in stream:
            line = line.strip()
            if not line.startswith("|"):
                continue
            fields = line.split("|")
            if len(fields) < 4 or not fields[1].isdigit():
                continue
            actions.append(_action_from_buttons(fields[2]))
    return actions


def parse_walkthrough(path: str) -> List[int]:
    suffix = os.path.splitext(path)[1].lower()
    if suffix == ".bk2":
        return parse_bk2_actions(path)
    if suffix == ".fm2":
        return parse_fm2_actions(path)
    raise ValueError(f"Unsupported walkthrough format: {path}")


def parse_walkthroughs(paths: Iterable[str]) -> List[int]:
    actions = []
    for path in paths:
        actions.extend(parse_walkthrough(path))
    active = sum(action != ACTION_NAMES.index("NOOP") for action in actions)
    if not actions or active == 0:
        raise ValueError("Walkthroughs contain no non-neutral controller actions")
    return actions
