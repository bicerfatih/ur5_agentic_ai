"""Episode storage for imitation-learning demonstrations."""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any

import numpy as np


@dataclass
class DemoTransition:
    obs: list[float]
    action: list[float]
    tcp_pose: list[float] = field(default_factory=list)
    target_xyz: list[float] = field(default_factory=list)
    label: str = ""
    timestamp: str = ""


@dataclass
class DemoEpisode:
    episode_id: str
    task: str
    transitions: list[DemoTransition]
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "episode_id": self.episode_id,
            "task": self.task,
            "meta": self.meta,
            "transitions": [asdict(t) for t in self.transitions],
        }


def default_demo_dir() -> str:
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.abspath(os.path.join(here, "../../data/demos/reach"))


def save_episode(episode: DemoEpisode, out_dir: str | None = None) -> str:
    root = out_dir or default_demo_dir()
    os.makedirs(root, exist_ok=True)
    path = os.path.join(root, f"{episode.episode_id}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(episode.to_dict(), f, indent=2)
    return path


def load_episodes(demo_dir: str | None = None) -> list[DemoEpisode]:
    root = demo_dir or default_demo_dir()
    if not os.path.isdir(root):
        return []
    episodes: list[DemoEpisode] = []
    for name in sorted(os.listdir(root)):
        if not name.endswith(".json"):
            continue
        path = os.path.join(root, name)
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
            transitions = [
                DemoTransition(**t) for t in data.get("transitions", []) if isinstance(t, dict)
            ]
            episodes.append(
                DemoEpisode(
                    episode_id=str(data.get("episode_id", name[:-5])),
                    task=str(data.get("task", "reach")),
                    transitions=transitions,
                    meta=dict(data.get("meta") or {}),
                )
            )
        except Exception:
            continue
    return episodes


def episodes_to_arrays(
    episodes: list[DemoEpisode],
) -> tuple[np.ndarray, np.ndarray]:
    xs: list[list[float]] = []
    ys: list[list[float]] = []
    for ep in episodes:
        for tr in ep.transitions:
            if len(tr.obs) >= 12 and len(tr.action) >= 3:
                xs.append(tr.obs[:12])
                ys.append(tr.action[:3])
    if not xs:
        return np.zeros((0, 12), dtype=np.float64), np.zeros((0, 3), dtype=np.float64)
    return np.asarray(xs, dtype=np.float64), np.asarray(ys, dtype=np.float64)


def new_episode_id(prefix: str = "demo") -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    return f"{prefix}_{stamp}"
