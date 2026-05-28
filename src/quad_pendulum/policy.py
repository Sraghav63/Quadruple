"""Policy-loading helpers shared by scripts."""

from __future__ import annotations

from pathlib import Path


def default_model_path(algo: str, links: int, explicit_model: Path | None = None) -> Path:
    """Return the explicit model path, or the best eval checkpoint when available."""
    if explicit_model is not None:
        return explicit_model

    run_name = f"{algo}_links{links}"
    best_model_path = Path("models") / f"{run_name}_best" / "best_model.zip"
    if best_model_path.exists():
        return best_model_path

    return Path("models") / f"{run_name}.zip"
