"""Run the full main-results pipeline for a single real-world dataset.

Given a config that points at a combined parquet (with transcription
columns), this script orchestrates every baseline and trained model
needed to fill the corresponding manuscript main-results table:

    1. Training-free baselines via :mod:`src.training.eval_baselines`
       (random, weighted_random, single_*, oracle).
    2. ROVER and weighted-ROVER via :mod:`src.training.rover`.
    3. Each trained variant in ``methods`` (e.g. ``mlp_pool``,
       ``hierarchical_transformer``) via :mod:`src.training.train`,
       followed by checkpoint evaluation on the test split.

Each step shells out to the relevant module so that this orchestrator
stays independent of training internals; the per-method results are
aggregated into a single ``main_results.json`` under ``--output-dir``.

Usage:
    python -m src.experiments.main_results \\
        --config configs/main_results_ami.json \\
        --output-dir reports/main_results/ami
"""

import json
import logging
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List

import typer

logger = logging.getLogger(__name__)
app = typer.Typer(help="Run the full main-results pipeline for one dataset.")


def _python_module(name: str, *args: str) -> List[str]:
    return [sys.executable, "-m", name, *args]


def _run(cmd: List[str], description: str) -> None:
    logger.info(">>> %s\n    %s", description, " ".join(cmd))
    completed = subprocess.run(cmd)
    if completed.returncode != 0:
        raise RuntimeError(
            f"{description} failed (exit {completed.returncode}): {' '.join(cmd)}"
        )


def _flags_from_dict(d: Dict[str, Any]) -> List[str]:
    out: List[str] = []
    for k, v in d.items():
        flag = f"--{k.replace('_', '-')}"
        if v is True:
            out.append(flag)
        elif v is False:
            continue
        else:
            out.extend([flag, str(v)])
    return out


@app.command()
def run(
    config: str = typer.Option(..., "--config", "-c"),
    output_dir: str = typer.Option(..., "--output-dir", "-o"),
    skip_training_free: bool = typer.Option(
        False, "--skip-training-free",
        help="Skip eval_baselines (use cached baselines.json).",
    ),
    skip_rover: bool = typer.Option(
        False, "--skip-rover", help="Skip ROVER (use cached rover.json).",
    ),
):
    """Run training-free + ROVER + every trained method, aggregate to JSON."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    cfg = json.loads(Path(config).read_text())
    parquet_path: str = cfg["parquet_path"]
    shared: Dict[str, Any] = cfg.get("shared", {})
    methods: List[Dict[str, Any]] = cfg.get("methods", [])

    out_root = Path(output_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    split_args = {
        k: shared[k]
        for k in ("train_ratio", "val_ratio", "seed")
        if k in shared
    }

    baselines_json = out_root / "baselines.json"
    if not skip_training_free:
        _run(_python_module(
            "src.training.eval_baselines",
            "--parquet-path", parquet_path,
            "--split", "test",
            "--save-json", str(baselines_json),
            *_flags_from_dict(split_args),
        ), description="training-free baselines")

    rover_json = out_root / "rover.json"
    if not skip_rover:
        _run(_python_module(
            "src.training.rover",
            "--parquet-path", parquet_path,
            "--split", "test",
            "--save-json", str(rover_json),
            *_flags_from_dict(split_args),
        ), description="ROVER baselines")

    aggregated: Dict[str, Any] = {
        "config_path": config,
        "parquet_path": parquet_path,
        "shared": shared,
        "training_free": (
            json.loads(baselines_json.read_text()) if baselines_json.exists() else None
        ),
        "rover": (
            json.loads(rover_json.read_text()) if rover_json.exists() else None
        ),
        "methods": {},
    }

    for method in methods:
        name = method["name"]
        logger.info("=" * 60)
        logger.info("Method: %s", name)
        logger.info("=" * 60)
        run_dir = out_root / name
        run_dir.mkdir(parents=True, exist_ok=True)
        merged = {**shared, **{k: v for k, v in method.items() if k != "name"}}

        train_flags = _flags_from_dict({
            "parquet-path": parquet_path,
            "experiment-name": name,
            "log-dir": str(out_root),
            **merged,
        })
        _run(_python_module("src.training.train", *train_flags),
             description=f"train method={name}")

        last_ckpt = out_root / name / "checkpoints" / "last.ckpt"
        test_json = run_dir / "test.json"
        eval_flags = [
            "--checkpoint", str(last_ckpt),
            "--parquet-path", parquet_path,
            "--split", "test",
            "--save-json", str(test_json),
        ]
        for opt in ("train_ratio", "val_ratio", "max_seq_len",
                    "batch_size", "num_workers", "seed"):
            if opt in merged:
                eval_flags += [f"--{opt.replace('_','-')}", str(merged[opt])]
        _run(_python_module("src.training.evaluate", *eval_flags),
             description=f"evaluate method={name}")

        aggregated["methods"][name] = {
            "config_overrides": {k: v for k, v in method.items() if k != "name"},
            "test": json.loads(test_json.read_text()),
        }
        with open(out_root / "main_results.json", "w") as f:
            json.dump(aggregated, f, indent=2)

    logger.info("Main-results pipeline complete. Results at %s",
                out_root / "main_results.json")


if __name__ == "__main__":
    app()
