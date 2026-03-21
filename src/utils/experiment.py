"""Experiment tracking helpers for reproducible pipeline runs."""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
from uuid import uuid4


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _safe_git_commit(cwd: Path) -> Optional[str]:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(cwd),
            stderr=subprocess.DEVNULL,
            text=True,
        )
        return out.strip() or None
    except Exception:
        return None


@dataclass
class ExperimentRun:
    """Container for one tracked experiment run."""

    run_id: str
    run_dir: Path
    started_at: str
    project_name: str
    mode: str
    config_path: str
    git_commit: Optional[str] = None
    artifacts: List[str] = field(default_factory=list)

    @property
    def manifest_path(self) -> Path:
        return self.run_dir / "manifest.json"

    @property
    def summary_path(self) -> Path:
        return self.run_dir / "summary.json"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "run_id": self.run_id,
            "run_dir": str(self.run_dir),
            "started_at": self.started_at,
            "project_name": self.project_name,
            "mode": self.mode,
            "config_path": self.config_path,
            "git_commit": self.git_commit,
            "artifacts": self.artifacts,
        }


def start_experiment_run(
    project_name: str,
    mode: str,
    config_path: str,
    output_dir: str = "runs",
    workspace_root: str = ".",
) -> ExperimentRun:
    """Initialize a tracked experiment run and write initial manifest."""
    start_time = _utc_now_iso()
    run_id = f"{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}-{uuid4().hex[:8]}"
    root = Path(output_dir)
    run_dir = root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    run = ExperimentRun(
        run_id=run_id,
        run_dir=run_dir,
        started_at=start_time,
        project_name=project_name,
        mode=mode,
        config_path=config_path,
        git_commit=_safe_git_commit(Path(workspace_root)),
    )
    write_json(run.manifest_path, run.to_dict())
    return run


def write_json(path: Path, payload: Dict[str, Any]) -> Path:
    """Write JSON payload to disk with stable formatting."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return path


def register_artifact(run: ExperimentRun, artifact_path: Path) -> None:
    """Register an artifact path in run manifest."""
    artifact_str = str(artifact_path)
    if artifact_str not in run.artifacts:
        run.artifacts.append(artifact_str)
        write_json(run.manifest_path, run.to_dict())


def finalize_experiment_run(
    run: ExperimentRun,
    status: str,
    metrics: Optional[Dict[str, Any]] = None,
    notes: Optional[str] = None,
) -> Path:
    """Write end-of-run summary for traceable experiment outcomes."""
    summary = {
        **run.to_dict(),
        "finished_at": _utc_now_iso(),
        "status": status,
        "metrics": metrics or {},
        "notes": notes or "",
    }
    return write_json(run.summary_path, summary)
