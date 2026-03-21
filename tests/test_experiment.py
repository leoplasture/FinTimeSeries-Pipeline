"""Unit tests for experiment tracking helpers."""

from __future__ import annotations

from pathlib import Path

from src.utils.experiment import (
    finalize_experiment_run,
    register_artifact,
    start_experiment_run,
)


def test_experiment_run_manifest_and_summary(tmp_path: Path) -> None:
    """Tracker should create manifest, register artifact, and write summary."""
    run = start_experiment_run(
        project_name="FinTimeSeries-Pipeline",
        mode="pipeline",
        config_path="config/params.yaml",
        output_dir=str(tmp_path / "runs"),
        workspace_root=str(tmp_path),
    )

    assert run.manifest_path.exists(), "Manifest should exist right after run start."

    artifact = run.run_dir / "artifacts" / "sample.csv"
    artifact.parent.mkdir(parents=True, exist_ok=True)
    artifact.write_text("a,b\n1,2\n", encoding="utf-8")

    register_artifact(run, artifact)
    summary_path = finalize_experiment_run(
        run, status="success", metrics={"rows": 10}, notes="unit test"
    )

    assert summary_path.exists(), "Summary file should be created at run completion."
    content = summary_path.read_text(encoding="utf-8")
    assert '"status": "success"' in content
    assert "sample.csv" in content
