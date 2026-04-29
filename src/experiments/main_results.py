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
import re
from pathlib import Path
from typing import Any, Dict, List

import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf

logger = logging.getLogger(__name__)


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


def _log_results_table(aggregated: Dict[str, Any]) -> None:
    """Print a nicely formatted table of the results for paper inclusion."""
    rows = []
    
    # 1. Training-free baselines
    if aggregated.get("training_free"):
        tf_data = aggregated["training_free"]
        # Extract WER metrics
        for k, v in tf_data.items():
            if isinstance(v, float) and "wer" in k.lower():
                name = k.replace("wer_", "").replace("_wer", "").replace("_", " ").title()
                # Format baselines similarly
                if isinstance(v, float):
                     rows.append((f"Baseline: {name}", v, 0.0, 1))
    
    # 2. ROVER baselines
    if aggregated.get("rover"):
        rover_data = aggregated["rover"]
        for k, v in rover_data.items():
            if isinstance(v, float) and "wer" in k.lower():
                name = k.replace("wer_", "").replace("_wer", "").replace("_", " ").title()
                rows.append((f"ROVER: {name}", v, 0.0, 1))

    # 3. Trained methods (grouped by seed)
    if aggregated.get("methods"):
        grouped_wers = {}
        for name, data in aggregated["methods"].items():
            wer = None
            if "test" in data and "selected_wer" in data["test"]:
                wer = data["test"]["selected_wer"]
            elif "test" in data and "wer" in data["test"]:
                wer = data["test"]["wer"]
            
            if wer is not None:
                # Group names by removing `-seed[num]` or `_seed[num]`
                base_name = re.sub(r"[-_]seed\d+$", "", name)
                grouped_wers.setdefault(base_name, []).append((name, wer))

        for base_name, items in grouped_wers.items():
            wers = [w for _, w in items]
            
            if len(wers) > 1:
                mean_wer = float(np.mean(wers))
                std_wer = float(np.std(wers))
                n = len(wers)
                rows.append((f"Model: {base_name} (mean)", mean_wer, std_wer, n))
                
                # Also append individual single-seed results
                for original_name, wer in items:
                    rows.append((f"Model: {original_name}", wer, 0.0, 1))
            else:
                original_name = items[0][0]
                rows.append((f"Model: {original_name}", wers[0], 0.0, 1))

    if not rows:
        logger.info("No results found to print.")
        return

    # Sort by WER ascending
    rows.sort(key=lambda x: x[1])

    logger.info("=" * 70)
    logger.info("FINAL EXPERIMENT RESULTS (TEST WER)")
    logger.info("=" * 70)
    logger.info(f"| {'Method':<40} | {'Test WER':<21} |")
    logger.info("-" * 70)
    for name, mean_wer, std_wer, n in rows:
        if n > 1:
            wer_str = f"{mean_wer:.4f} ± {std_wer:.4f} (n={n})"
        else:
            wer_str = f"{mean_wer:.4f}              "
        logger.info(f"| {name:<40} | {wer_str:<21} |")
    logger.info("=" * 70)


@hydra.main(version_base="1.3", config_path="../../configs", config_name="orchestrator")
def run(cfg: DictConfig):
    """Run training-free + ROVER + every trained method, aggregate to JSON."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    pipeline = cfg.pipeline
    parquet_path: str = pipeline.parquet_path
    shared = OmegaConf.to_container(pipeline.get("shared", {}), resolve=True)
    raw_methods = OmegaConf.to_container(pipeline.get("methods", []), resolve=True)

    # Automatically expand methods with multiple seeds
    methods = []
    for method in raw_methods:
        seeds = method.get("seed", None)
        if isinstance(seeds, list):
            for s in seeds:
                new_method = dict(method)
                new_method["seed"] = s
                new_method["name"] = f"{method['name']}-seed{s}"
                methods.append(new_method)
        else:
            methods.append(method)

    out_root = Path(cfg.get("output_dir", "reports/main_results/ami"))
    out_root.mkdir(parents=True, exist_ok=True)

    split_args = {
        k: shared[k]
        for k in ("train_ratio", "val_ratio", "seed")
        if k in shared
    }

    baselines_json = out_root / "baselines.json"
    if not cfg.get("skip_training_free", False):
        _run(_python_module(
            "src.training.eval_baselines",
            "--parquet-path", parquet_path,
            "--split", "test",
            "--save-json", str(baselines_json),
            *_flags_from_dict(split_args),
        ), description="training-free baselines")

    rover_json = out_root / "rover.json"
    if not cfg.get("skip_rover", False):
        _run(_python_module(
            "src.training.rover",
            "--parquet-path", parquet_path,
            "--split", "test",
            "--save-json", str(rover_json),
            *_flags_from_dict(split_args),
        ), description="ROVER baselines")

    aggregated: Dict[str, Any] = {
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

        # Format as Hydra overrides instead of Typer flags
        train_flags = [
            f"parquet_path={parquet_path}",
            f"experiment_name={name}",
            f"log_dir={out_root}"
        ]
        for k, v in merged.items():
            if v is True:
                train_flags.append(f"{k}=true")
            elif v is False:
                train_flags.append(f"{k}=false")
            elif v is not None:
                train_flags.append(f"{k}={v}")
        _run(_python_module("src.training.train", *train_flags),
             description=f"train method={name}")

        # train.py runs trainer.test on the best checkpoint and writes
        # test_results.json into the experiment dir, so we just read it.
        test_json = out_root / name / "test_results.json"
        aggregated["methods"][name] = {
            "config_overrides": {k: v for k, v in method.items() if k != "name"},
            "test": json.loads(test_json.read_text()),
        }
        with open(out_root / "main_results.json", "w") as f:
            json.dump(aggregated, f, indent=2)

    _log_results_table(aggregated)

    logger.info("Main-results pipeline complete. Results at %s",
                out_root / "main_results.json")


if __name__ == "__main__":
    run()
