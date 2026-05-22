# config/programs.py — whitelist helpers for .urp programs

from config.settings import ALLOWED_URP_PROGRAMS


def normalize_urp_name(name: str) -> str:
    name = (name or "").strip()
    if not name:
        return ""
    return name if name.endswith(".urp") else f"{name}.urp"


def is_program_allowed(program_name: str) -> bool:
    """True if program is on the configured whitelist."""
    if not program_name:
        return False
    raw = program_name.strip()
    normalized = normalize_urp_name(raw)
    base = normalized.replace(".urp", "")
    allowed = set(ALLOWED_URP_PROGRAMS)
    return raw in allowed or normalized in allowed or base in allowed
