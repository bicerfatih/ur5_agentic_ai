"""Train a lightweight behavior-cloning policy from recorded demos."""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from il.demo_store import episodes_to_arrays, load_episodes


def _ridge_fit(X: np.ndarray, Y: np.ndarray, alpha: float = 1e-3) -> tuple[np.ndarray, np.ndarray]:
    """Multi-output ridge: Y ≈ X @ W + b."""
    n, d = X.shape
    ones = np.ones((n, 1), dtype=np.float64)
    Xb = np.hstack([X, ones])
    k = Xb.shape[1]
    reg = alpha * np.eye(k, dtype=np.float64)
    reg[-1, -1] = 0.0
    w_full = np.linalg.solve(Xb.T @ Xb + reg, Xb.T @ Y)
    W = w_full[:-1, :]
    b = w_full[-1, :]
    return W, b


def train_bc(
    demo_dir: str,
    out_dir: str,
    alpha: float = 1e-3,
    val_ratio: float = 0.2,
) -> dict:
    episodes = load_episodes(demo_dir)
    X, Y = episodes_to_arrays(episodes)
    if X.shape[0] < 8:
        raise RuntimeError(f"Need at least 8 transitions; found {X.shape[0]} in {demo_dir}")

    n = X.shape[0]
    idx = np.arange(n)
    rng = np.random.default_rng(42)
    rng.shuffle(idx)
    n_val = max(1, int(n * val_ratio))
    val_idx, train_idx = idx[:n_val], idx[n_val:]
    if train_idx.size < 4:
        train_idx, val_idx = idx, idx[:1]

    W, b = _ridge_fit(X[train_idx], Y[train_idx], alpha=alpha)
    pred_train = X[train_idx] @ W + b
    pred_val = X[val_idx] @ W + b
    train_mae = float(np.mean(np.abs(pred_train - Y[train_idx])))
    val_mae = float(np.mean(np.abs(pred_val - Y[val_idx])))

    os.makedirs(out_dir, exist_ok=True)
    np.savez(
        os.path.join(out_dir, "bc_weights.npz"),
        W=W.astype(np.float32),
        b=b.astype(np.float32),
    )
    meta = {
        "policy_type": "bc_ridge",
        "demo_dir": os.path.abspath(demo_dir),
        "n_episodes": len(episodes),
        "n_transitions": int(n),
        "train_mae_m": round(train_mae, 6),
        "val_mae_m": round(val_mae, 6),
        "obs_dim": 12,
        "action_dim": 3,
        "alpha": alpha,
    }
    with open(os.path.join(out_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    return meta


def main():
    p = argparse.ArgumentParser(description="Train behavior cloning from recorded reach demos.")
    p.add_argument("--demo-dir", default="", help="Directory with demo_*.json episodes")
    p.add_argument("--task-id", default="reach_bc")
    p.add_argument("--version", default="v1")
    p.add_argument("--alpha", type=float, default=1e-3, help="Ridge regularization")
    args = p.parse_args()

    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    demo_dir = args.demo_dir or os.path.join(here, "../data/demos/reach")
    out_dir = os.path.join(here, f"../data/models/policies/{args.task_id}_{args.version}")
    meta = train_bc(demo_dir=demo_dir, out_dir=out_dir, alpha=args.alpha)
    print(json.dumps({"status": "done", "out_dir": os.path.abspath(out_dir), **meta}, indent=2))


if __name__ == "__main__":
    main()
