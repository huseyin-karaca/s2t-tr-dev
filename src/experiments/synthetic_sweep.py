"""Sweep the synthetic regime-switch experiment over the number of regimes.

This script orchestrates the full pipeline (data generation → router
training → checkpoint evaluation → training-free baselines) for a list
of ``--r-values``. Per-R results are accumulated into a JSON sweep
file, and a separate ``figure`` subcommand renders the final
manuscript-ready plot from that JSON.

Each step shells out to the existing CLI entry points
(:mod:`src.data.synthetic`, :mod:`src.training.train`,
:mod:`src.training.evaluate`, :mod:`src.training.eval_baselines`) via
``subprocess`` so that no module here has to know about training
internals; this also keeps the per-run process state isolated.

Usage:
    python -m src.experiments.synthetic_sweep run \\
        --r-values 2,3,4,6,8 \\
        --output-dir reports/sweeps/synthetic_R \\
        --num-samples 5000 --frame-length 128 \\
        --max-epochs 30 --batch-size 32 --seed 42

    python -m src.experiments.synthetic_sweep figure \\
        --results reports/sweeps/synthetic_R/sweep_results.json \\
        --output-path reports/manuscript/figures/synthetic_sweep.pdf
"""

import json
import logging
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import typer

logger = logging.getLogger(__name__)
app = typer.Typer(help="Sweep the synthetic regime-switch experiment over R.")


# ---------------------------------------------------------------------------
# run subcommand
# ---------------------------------------------------------------------------


@dataclass
class _RunPaths:
    """Paths produced by a single sweep cell (one R, one architecture).

    ``test_json`` is the file that ``src.training.train`` writes after
    its in-fit ``trainer.test`` (on the best checkpoint). The orchestrator
    just reads it instead of running a second evaluate subprocess.
    """
    parquet:    Path
    log_dir:    Path
    test_json:  Path = field(init=False)

    def __post_init__(self):
        self.test_json = self.log_dir / "test_results.json"


def _run(cmd: List[str], description: str) -> None:
    """Run ``cmd`` synchronously and stream its output to the parent stderr."""
    logger.info(">>> %s\n    %s", description, " ".join(cmd))
    completed = subprocess.run(cmd)
    if completed.returncode != 0:
        raise RuntimeError(f"{description} failed (exit {completed.returncode}): {' '.join(cmd)}")


def _python_module(name: str, *args: str) -> List[str]:
    """Build a ``[python, -m, name, *args]`` invocation that uses the same interpreter."""
    return [sys.executable, "-m", name, *args]


def _generate_dataset(
    parquet_path: Path,
    num_samples: int,
    frame_length: int,
    num_regimes: int,
    regime_dim: int,
    noise_std: float,
    feature_dtype: str,
    write_batch_size: int,
    seed: int,
) -> None:
    """Generate one synthetic parquet via :mod:`src.data.synthetic`."""
    cmd = _python_module(
        "src.data.synthetic",
        "--output-path",     str(parquet_path),
        "--num-samples",     str(num_samples),
        "--frame-length",    str(frame_length),
        "--num-regimes",     str(num_regimes),
        "--regime-dim",      str(regime_dim),
        "--noise-std",       str(noise_std),
        "--feature-dtype",   feature_dtype,
        "--write-batch-size", str(write_batch_size),
        "--seed",            str(seed),
    )
    _run(cmd, f"generate synthetic parquet (R={num_regimes})")


def _train_router(
    parquet: Path,
    log_dir: Path,
    arch: str,
    max_seq_len: int,
    batch_size: int,
    num_workers: int,
    max_epochs: int,
    learning_rate: float,
    primary_weight: float,
    aux_ce_weight: float,
    soft_ce_weight: float,
    soft_ce_temperature: float,
    seed: int,
) -> None:
    """Train one router via :mod:`src.training.train`."""
    cmd = _python_module(
        "src.training.train",
        "--parquet-path",         str(parquet),
        "--arch",                 arch,
        "--max-seq-len",          str(max_seq_len),
        "--batch-size",           str(batch_size),
        "--num-workers",          str(num_workers),
        "--max-epochs",           str(max_epochs),
        "--learning-rate",        str(learning_rate),
        "--primary-weight",       str(primary_weight),
        "--aux-ce-weight",        str(aux_ce_weight),
        "--soft-ce-weight",       str(soft_ce_weight),
        "--soft-ce-temperature",  str(soft_ce_temperature),
        "--seed",                 str(seed),
        "--experiment-name",      log_dir.name,
        "--log-dir",              str(log_dir.parent),
    )
    _run(cmd, f"train router (arch={arch}, log_dir={log_dir})")


def _evaluate_baselines(
    parquet: Path,
    save_json: Path,
    train_ratio: float,
    val_ratio: float,
    seed: int,
) -> None:
    """Compute training-free baselines via :mod:`src.training.eval_baselines`."""
    cmd = _python_module(
        "src.training.eval_baselines",
        "--parquet-path", str(parquet),
        "--train-ratio",  str(train_ratio),
        "--val-ratio",    str(val_ratio),
        "--seed",         str(seed),
        "--split",        "test",
        "--save-json",    str(save_json),
    )
    _run(cmd, f"eval_baselines on {parquet}")


@app.command()
def run(
    output_dir: str = typer.Option(..., "--output-dir", "-o"),
    r_values: str = typer.Option(
        "2,3,4,6,8", "--r-values",
        help="Comma-separated regime counts to sweep over.",
    ),
    num_samples: int = typer.Option(5000, "--num-samples"),
    frame_length: int = typer.Option(128, "--frame-length"),
    regime_dim: int = typer.Option(32, "--regime-dim"),
    noise_std: float = typer.Option(0.5, "--noise-std"),
    feature_dtype: str = typer.Option("float16", "--feature-dtype"),
    write_batch_size: int = typer.Option(500, "--write-batch-size"),
    train_ratio: float = typer.Option(0.8, "--train-ratio"),
    val_ratio: float = typer.Option(0.1, "--val-ratio"),
    max_seq_len: int = typer.Option(256, "--max-seq-len"),
    batch_size: int = typer.Option(32, "--batch-size"),
    num_workers: int = typer.Option(2, "--num-workers"),
    max_epochs: int = typer.Option(30, "--max-epochs"),
    learning_rate: float = typer.Option(1e-4, "--learning-rate", "--lr"),
    primary_weight: float = typer.Option(1.0, "--primary-weight"),
    aux_ce_weight: float = typer.Option(0.0, "--aux-ce-weight"),
    soft_ce_weight: float = typer.Option(0.5, "--soft-ce-weight"),
    soft_ce_temperature: float = typer.Option(0.1, "--soft-ce-temperature"),
    seed: int = typer.Option(42, "--seed"),
    keep_parquets: bool = typer.Option(
        False, "--keep-parquets/--no-keep-parquets",
        help="If false, delete each per-R parquet once that R cell is done "
             "(saves disk; defaults to off).",
    ),
):
    """Generate, train, and evaluate the full pipeline for each R value."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    r_list = [int(x) for x in r_values.split(",") if x.strip()]
    logger.info("Sweep R values: %s — output dir: %s", r_list, out_dir)

    sweep_results: Dict = {
        "config": {
            "r_values":            r_list,
            "num_samples":         num_samples,
            "frame_length":        frame_length,
            "regime_dim":          regime_dim,
            "noise_std":           noise_std,
            "max_seq_len":         max_seq_len,
            "batch_size":          batch_size,
            "max_epochs":          max_epochs,
            "learning_rate":       learning_rate,
            "primary_weight":      primary_weight,
            "aux_ce_weight":       aux_ce_weight,
            "soft_ce_weight":      soft_ce_weight,
            "soft_ce_temperature": soft_ce_temperature,
            "train_ratio":         train_ratio,
            "val_ratio":           val_ratio,
            "seed":                seed,
        },
        "by_r": {},
    }

    for R in r_list:
        cell_dir = out_dir / f"R{R}"
        cell_dir.mkdir(parents=True, exist_ok=True)
        parquet = cell_dir / "combined_features.parquet"
        hier_paths = _RunPaths(
            parquet=parquet,
            log_dir=cell_dir / "hier",
        )
        mlp_paths = _RunPaths(
            parquet=parquet,
            log_dir=cell_dir / "mlp_pool",
        )
        baselines_json = cell_dir / "baselines.json"

        _generate_dataset(
            parquet_path=parquet,
            num_samples=num_samples,
            frame_length=frame_length,
            num_regimes=R,
            regime_dim=regime_dim,
            noise_std=noise_std,
            feature_dtype=feature_dtype,
            write_batch_size=write_batch_size,
            seed=seed,
        )
        for paths, arch in [(hier_paths, "hierarchical_transformer"), (mlp_paths, "mlp_pool")]:
            _train_router(
                parquet=parquet,
                log_dir=paths.log_dir,
                arch=arch,
                max_seq_len=max_seq_len,
                batch_size=batch_size,
                num_workers=num_workers,
                max_epochs=max_epochs,
                learning_rate=learning_rate,
                primary_weight=primary_weight,
                aux_ce_weight=aux_ce_weight,
                soft_ce_weight=soft_ce_weight,
                soft_ce_temperature=soft_ce_temperature,
                seed=seed,
            )
            # train.py writes test_results.json into paths.log_dir after
            # its in-fit trainer.test on the best checkpoint, so no
            # second subprocess is needed.
        _evaluate_baselines(
            parquet=parquet,
            save_json=baselines_json,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            seed=seed,
        )

        sweep_results["by_r"][str(R)] = {
            "hier_test":    json.loads(hier_paths.test_json.read_text()),
            "mlp_pool_test": json.loads(mlp_paths.test_json.read_text()),
            "baselines":    json.loads(baselines_json.read_text()),
        }
        with open(out_dir / "sweep_results.json", "w") as f:
            json.dump(sweep_results, f, indent=2)
        logger.info("Sweep cell R=%d complete.", R)

        if not keep_parquets and parquet.exists():
            parquet.unlink()
            logger.info("Removed parquet %s to save disk; pass --keep-parquets to retain.", parquet)

    logger.info("All sweep cells complete; results at %s", out_dir / "sweep_results.json")


# ---------------------------------------------------------------------------
# figure subcommand
# ---------------------------------------------------------------------------


def _series_for_baseline(by_r: Dict, baseline_name: str) -> List[Optional[float]]:
    """Pull the test-split mean WER of a training-free baseline at each R."""
    out: List[Optional[float]] = []
    for R, entry in sorted(by_r.items(), key=lambda kv: int(kv[0])):
        results = entry["baselines"]["results"]["test"]
        out.append(float(results[baseline_name]["wer_mean"]) if baseline_name in results else None)
    return out


def _series_for_router(by_r: Dict, key: str) -> List[Optional[float]]:
    """Pull a metric (e.g. ``selected_wer``) from each R's evaluate.py JSON."""
    out: List[Optional[float]] = []
    for R, entry in sorted(by_r.items(), key=lambda kv: int(kv[0])):
        v = entry.get(key, {}).get("selected_wer")
        out.append(float(v) if v is not None else None)
    return out


@app.command()
def figure(
    results: str = typer.Option(..., "--results", "-r"),
    output_path: str = typer.Option(..., "--output-path", "-o"),
):
    """Render the manuscript sweep figure from a sweep_results.json file."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    data = json.loads(Path(results).read_text())
    by_r = data["by_r"]
    r_values = sorted([int(R) for R in by_r.keys()])

    series = {
        "Random":           _series_for_baseline(by_r, "random"),
        "Weighted random":  _series_for_baseline(by_r, "weighted_random"),
        "MLP-pool":         _series_for_router(by_r, "mlp_pool_test"),
        "Proposed":         _series_for_router(by_r, "hier_test"),
        "Oracle":           _series_for_baseline(by_r, "oracle"),
    }

    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    style = {
        "Random":          dict(color="#888888", linestyle=":",  marker="o"),
        "Weighted random": dict(color="#555555", linestyle="--", marker="s"),
        "MLP-pool":        dict(color="#d62728", linestyle="-",  marker="^"),
        "Proposed":        dict(color="#1f77b4", linestyle="-",  marker="D", linewidth=2.0),
        "Oracle":          dict(color="#2ca02c", linestyle="-.", marker="x"),
    }
    for name, ys in series.items():
        ax.plot(r_values, ys, label=name, markersize=6, **style.get(name, {}))
    ax.set_xlabel("Number of regimes $R$")
    ax.set_ylabel("Test WER")
    ax.set_xticks(r_values)
    ax.grid(alpha=0.3)
    ax.legend(loc="best", frameon=True)
    ax.set_title("Synthetic regime-switch sweep — test WER vs. number of regimes")
    fig.tight_layout()

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    if out.suffix.lower() == ".pdf":
        fig.savefig(out.with_suffix(".png"), bbox_inches="tight", dpi=150)
    plt.close(fig)
    logger.info("Wrote sweep figure to %s", out)


if __name__ == "__main__":
    app()
