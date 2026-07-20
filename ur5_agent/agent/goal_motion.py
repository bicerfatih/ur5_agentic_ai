# agent/goal_motion.py — parse spoken/typed Cartesian goals into move_* tool calls

import re

_CARTESIAN_TOOLS = frozenset(
    {
        "move_up",
        "move_down",
        "move_left",
        "move_right",
        "move_forward",
        "move_backward",
    }
)

_DIR_TOOLS = {
    "left": "move_left",
    "right": "move_right",
    "up": "move_up",
    "down": "move_down",
    "forward": "move_forward",
    "forwards": "move_forward",
    "back": "move_backward",
    "backward": "move_backward",
    "backwards": "move_backward",
}

_UNIT = r"(cm|centimeters?|mm|millimeters?|m|meters?)"
_DIR = r"(left|right|up|down|forward|forwards?|back(?:ward)?s?)"
_AMOUNT = r"(\d+(?:\.\d+)?)"

_PATTERNS = (
    re.compile(
        rf"(?:move|go)\s+{_DIR}\s+(?:for\s+)?{_AMOUNT}\s*{_UNIT}\b",
        re.IGNORECASE,
    ),
    re.compile(
        rf"{_AMOUNT}\s*{_UNIT}\s+{_DIR}\b",
        re.IGNORECASE,
    ),
)

_COMPOUND = re.compile(
    r"\b(and|then|also|after|before|twice|again|gripper|open|close|detect|camera|home|state|pick|place)\b",
    re.IGNORECASE,
)


def _normalize_goal(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip().lower()).strip(".,!?;:")


def _length_to_meters(amount: str, unit: str) -> float:
    n = float(amount)
    u = unit.lower().rstrip("s")
    if u in ("cm", "centimeter"):
        return n / 100.0
    if u in ("mm", "millimeter"):
        return n / 1000.0
    if u in ("m", "meter"):
        return n
    raise ValueError(f"unknown unit: {unit}")


def _tool_for_direction(direction: str) -> str | None:
    return _DIR_TOOLS.get(direction.lower())


def parse_cartesian_motion_goal(goal: str) -> list[dict] | None:
    """
    Map a simple goal like "move left 20 cm" to one coordinated Cartesian tool call.
    Returns None when the goal is compound or not a single distance + direction.
    """
    text = _normalize_goal(goal)
    if not text or _COMPOUND.search(text):
        return None

    for pattern in _PATTERNS:
        match = pattern.search(text)
        if not match:
            continue
        groups = match.groups()
        if len(groups) == 3 and groups[0] in _DIR_TOOLS:
            direction, amount, unit = groups
        else:
            amount, unit, direction = groups
        tool = _tool_for_direction(direction)
        if not tool or tool not in _CARTESIAN_TOOLS:
            return None
        try:
            distance_m = _length_to_meters(amount, unit)
        except ValueError:
            return None
        if distance_m <= 0:
            return None
        return [{"name": tool, "arguments": {"distance_m": round(distance_m, 4)}}]

    return None
