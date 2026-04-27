"""Run a JSON-driven training ablation over a single feature parquet.

Each entry in the config's ``variants`` list is one training run; the
fields are passed straight to :mod:`src.training.train` as ``--name
value`` flags. After every run we evaluate the best checkpoint on the
test split via :mod:`src.training.evaluate` and append the result to a
single ``ablation_results.json``.

This script is generic: any flag accepted by ``train.py`` can be set
in a variant. The driving config (e.g. ``configs/architecture_ablation.json``)
declares the parquet path, shared training options, and the per-variant
overrides; it is intentionally schema-light to keep ablations cheap to
add.

Usage:
    python -m src.experiments.run_ablation \\
        --config configs/architecture_ablation.json \\
        --output-dir reports/ablations/architecture_ami
"""

import json
import logging
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List

import typer

logger = logging.getLogger(__name__)
app = typer.Typer(help="Run a JSON-driven training ablation.")


def _python_module(name: str, *args: str) -> List[str]:
    return [sys.executable, "-m", name, *args]


def _flags_from_dict(d: Dict[str, Any]) -> List[str]:
    """Convert a dict of ``{flag_name: value}`` to a typer-compatible flag list.

    Booleans become ``--flag`` / ``--no-flag`` style toggles only when
    the value is the literal sentinel ``"FLAG_TRUE"`` / ``"FLAG_FALSE"``;
    plain ``True/False`` are passed through as strings to keep this
    helper boring and predictable. Values may be int/float/str.
    """
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


def _run(cmd: List[str], description: str) -> None:
    logger.info(">>> %s\n    %s", description, " ".join(cmd))
    completed = subprocess.run(cmd)
    if completed.returncode != 0:
        raise RuntimeError(
            f"{description} failed (exit {completed.returncode}): {' '.join(cmd)}"
        )


@app.command()
def run(
    config: str = typer.Option(..., "--config", "-c"),
    output_dir: str = typer.Option(..., "--output-dir", "-o"),
):
    """Run every variant in ``config`` and dump a combined results JSON."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    cfg = json.loads(Path(config).read_text())
    parquet_path: str = cfg["parquet_path"]
    shared: Dict[str, Any] = cfg.get("shared", {})
    variants: List[Dict[str, Any]] = cfg["variants"]

    out_root = Path(output_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    results: Dict[str, Any] = {
        "config_path": config,
        "parquet_path": parquet_path,
        "shared": shared,
        "variants": {},
    }

    for variant in variants:
        name = variant["name"]
        logger.info("=" * 60)
        logger.info("Variant: %s", name)
        logger.info("=" * 60)
        run_dir = out_root / name
        run_dir.mkdir(parents=True, exist_ok=True)

        merged = {**shared, **{k: v for k, v in variant.items() if k != "name"}}
        train_flags = _flags_from_dict({
            "parquet-path": parquet_path,
            "experiment-name": name,
            "log-dir": str(out_root),
            **merged,
        })
        _run(_python_module("src.training.train", *train_flags),
             description=f"train variant={name}")

        # train.py runs trainer.test on the best checkpoint and dumps
        # test_results.json — read it directly, no second subprocess.
        test_json = out_root / name / "test_results.json"
        results["variants"][name] = {
            "config_overrides": {k: v for k, v in variant.items() if k != "name"},
            "test": json.loads(test_json.read_text()),
        }
        with open(out_root / "ablation_results.json", "w") as f:
            json.dump(results, f, indent=2)

    logger.info("Ablation complete. Aggregated results at %s",
                out_root / "ablation_results.json")


if __name__ == "__main__":
    app()
