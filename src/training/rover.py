"""ROVER and weighted-ROVER hypothesis-combination baselines.

ROVER (Recognizer Output Voting Error Reduction) combines hypotheses
from multiple ASR systems by aligning them word-by-word and voting at
each aligned position. The implementation here is a self-contained
progressive multi-alignment of word strings followed by per-slot
majority voting; it is sufficient for reporting ROVER as a baseline
in the manuscript and works on any combined parquet that carries the
``{whisper,hubert,w2v2}_transcription`` and ``{whisper,hubert,w2v2}_wer``
columns produced by :mod:`src.data.preprocess`.

Two variants are reported:
    rover (uniform): each system contributes one vote per slot.
    weighted_rover: each system's vote is weighted by ``1 - WER_train``
        (its training-split accuracy proxy), normalized to sum to one.

Usage:
    python -m src.training.rover \\
        --parquet-path data/processed/edinburghcstr_ami/combined_features_with_transcripts.parquet \\
        --save-json logs/edinburghcstr_ami/rover.json
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import jiwer
import numpy as np
import pyarrow.parquet as pq
import typer

from src.data.dataset import MODEL_NAMES, WER_COLUMNS

logger = logging.getLogger(__name__)
app = typer.Typer(help="ROVER and weighted-ROVER hypothesis combination baselines.")

TRANSCRIPTION_COLUMNS: Dict[str, str] = {
    "hubert":   "hubert_transcription",
    "whisper":  "whisper_transcription",
    "wav2vec2": "w2v2_transcription",
}

NULL = ""  # sentinel for an empty alignment slot (gap)


def _split_indices(
    n_total: int, train_ratio: float, val_ratio: float, seed: int,
) -> Dict[str, np.ndarray]:
    """Mirror the train/val/test split used by :mod:`src.training.train`."""
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)
    perm = np.random.default_rng(seed).permutation(n_total)
    return {
        "train": perm[:n_train],
        "val": perm[n_train:n_train + n_val],
        "test": perm[n_train + n_val:],
    }


def _pairwise_align(
    a: List[List[str]], b: List[str],
) -> List[List[str]]:
    """Align a list-of-slots ``a`` to a flat token list ``b`` via Levenshtein DP.

    ``a`` is a confusion network represented as a list of slots, where
    each slot is a list of tokens (one per system already merged in).
    ``b`` is a flat token list (the next system's hypothesis). The
    returned confusion network has the same length as the alignment
    path, with ``b``'s token appended to the matched slot, or a new
    slot containing ``b``'s token (with NULLs for the systems already
    in ``a``) when ``b`` has an insertion, or ``a``'s slot extended by
    a NULL when ``b`` has a deletion.

    Args:
        a: Existing confusion network. Each slot is a list of strings.
        b: New hypothesis to align.

    Returns:
        Combined confusion network with one extra column per slot.
    """
    n, m = len(a), len(b)
    n_systems = len(a[0]) if a else 0
    dp = np.zeros((n + 1, m + 1), dtype=np.int32)
    for i in range(n + 1):
        dp[i, 0] = i
    for j in range(m + 1):
        dp[0, j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            slot_match = b[j - 1] in a[i - 1]
            sub_cost = 0 if slot_match else 1
            dp[i, j] = min(
                dp[i - 1, j - 1] + sub_cost,
                dp[i - 1, j] + 1,
                dp[i, j - 1] + 1,
            )

    out: List[List[str]] = []
    i, j = n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0:
            slot_match = b[j - 1] in a[i - 1]
            sub_cost = 0 if slot_match else 1
            if dp[i, j] == dp[i - 1, j - 1] + sub_cost:
                out.append(a[i - 1] + [b[j - 1]])
                i -= 1; j -= 1
                continue
        if i > 0 and dp[i, j] == dp[i - 1, j] + 1:
            out.append(a[i - 1] + [NULL])
            i -= 1
            continue
        out.append([NULL] * n_systems + [b[j - 1]])
        j -= 1
    out.reverse()
    return out


def _build_confusion_network(hypotheses: List[List[str]]) -> List[List[str]]:
    """Progressively align ``len(hypotheses)`` token lists into a CN.

    The first hypothesis seeds the network as a column of single-token
    slots; each subsequent hypothesis is aligned to the running CN.
    The final CN has one column per system, in input order.
    """
    if not hypotheses:
        return []
    cn: List[List[str]] = [[tok] for tok in hypotheses[0]]
    for h in hypotheses[1:]:
        cn = _pairwise_align(cn, h)
    return cn


def _vote(cn: List[List[str]], weights: np.ndarray) -> List[str]:
    """Per-slot weighted majority vote over a confusion network.

    Args:
        cn: Confusion network with shape ``(num_slots, num_systems)``.
        weights: 1-D array of per-system weights, summing to 1.

    Returns:
        The voted token list (NULLs filtered out).
    """
    out: List[str] = []
    for slot in cn:
        scores: Dict[str, float] = {}
        for tok, w in zip(slot, weights):
            scores[tok] = scores.get(tok, 0.0) + float(w)
        best = max(scores.items(), key=lambda kv: kv[1])[0]
        if best != NULL:
            out.append(best)
    return out


def _rover_one(hypotheses: List[str], weights: np.ndarray) -> str:
    """Run one ROVER pass over ``hypotheses`` (in :data:`MODEL_NAMES` order)."""
    token_lists = [h.split() for h in hypotheses]
    cn = _build_confusion_network(token_lists)
    return " ".join(_vote(cn, weights)).strip()


def _wer_corpus(refs: List[str], hyps: List[str]) -> float:
    """Corpus-level WER via :mod:`jiwer`."""
    refs2 = [(r if r.strip() else "<empty>") for r in refs]
    hyps2 = [(h if h.strip() else "") for h in hyps]
    return float(jiwer.wer(refs2, hyps2))


def _wer_per_clip(refs: List[str], hyps: List[str]) -> np.ndarray:
    """Per-clip WER vector (length ``len(refs)``)."""
    out = np.zeros(len(refs), dtype=np.float32)
    for i, (r, h) in enumerate(zip(refs, hyps)):
        if not r.strip():
            out[i] = 0.0 if not h.strip() else 1.0
        else:
            out[i] = float(jiwer.wer(r, h))
    return out


def _summarize(per_clip: np.ndarray, corpus: float) -> Dict[str, float]:
    n = per_clip.shape[0]
    sem = float(per_clip.std(ddof=1) / np.sqrt(max(n, 1))) if n > 1 else 0.0
    return {
        "wer_mean": float(per_clip.mean()),
        "wer_sem": sem,
        "wer_corpus": float(corpus),
        "n": int(n),
    }


def _load_columns(parquet_path: str) -> Tuple[List[str], Dict[str, List[str]], np.ndarray]:
    """Load ground-truth, per-system transcriptions, and per-system WER."""
    cols = (
        ["ground_truth"]
        + [TRANSCRIPTION_COLUMNS[n] for n in MODEL_NAMES]
        + [WER_COLUMNS[n] for n in MODEL_NAMES]
    )
    table = pq.read_table(parquet_path, columns=cols)
    refs = [str(s) for s in table["ground_truth"].to_pylist()]
    hyps = {
        n: [str(s) for s in table[TRANSCRIPTION_COLUMNS[n]].to_pylist()]
        for n in MODEL_NAMES
    }
    wer = np.stack(
        [table[WER_COLUMNS[n]].to_numpy().astype(np.float32) for n in MODEL_NAMES],
        axis=-1,
    )
    return refs, hyps, wer


def _run_rover(
    refs: List[str],
    hyps: Dict[str, List[str]],
    indices: np.ndarray,
    weights: np.ndarray,
) -> Tuple[List[str], np.ndarray, float]:
    """Run ROVER over ``indices`` with given system weights."""
    sub_refs = [refs[i] for i in indices]
    rover_hyps = [
        _rover_one([hyps[n][i] for n in MODEL_NAMES], weights)
        for i in indices
    ]
    per_clip = _wer_per_clip(sub_refs, rover_hyps)
    corpus = _wer_corpus(sub_refs, rover_hyps)
    return rover_hyps, per_clip, corpus


@app.command()
def evaluate(
    parquet_path: str = typer.Option(..., "--parquet-path", "-p"),
    train_ratio: float = typer.Option(0.8, "--train-ratio"),
    val_ratio: float = typer.Option(0.1, "--val-ratio"),
    seed: int = typer.Option(42, "--seed"),
    split: str = typer.Option("test", "--split"),
    save_json: Optional[str] = typer.Option(None, "--save-json"),
):
    """Compute ROVER and weighted-ROVER on the requested split."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    refs, hyps, wer = _load_columns(parquet_path)
    n_total = len(refs)
    splits = _split_indices(n_total, train_ratio, val_ratio, seed)
    logger.info(
        "Loaded %d clips from %s — train=%d val=%d test=%d",
        n_total, parquet_path,
        len(splits["train"]), len(splits["val"]), len(splits["test"]),
    )

    if split == "all":
        target = ["train", "val", "test"]
    elif split in splits:
        target = [split]
    else:
        raise ValueError(f"Unknown split={split!r}; expected train/val/test/all")

    train_wer_mean = wer[splits["train"]].mean(axis=0)
    inv = np.maximum(1.0 - train_wer_mean, 1e-6)
    weighted = inv / inv.sum()
    uniform = np.full(len(MODEL_NAMES), 1.0 / len(MODEL_NAMES), dtype=np.float32)
    logger.info("Train-WER means: %s", dict(zip(MODEL_NAMES, train_wer_mean.tolist())))
    logger.info("Weighted-ROVER weights: %s", dict(zip(MODEL_NAMES, weighted.tolist())))

    results: Dict[str, Dict] = {}
    for s in target:
        idx = splits[s]
        per_split: Dict[str, Dict] = {}
        for name, w in [("rover", uniform), ("weighted_rover", weighted)]:
            _, per_clip, corpus = _run_rover(refs, hyps, idx, w)
            per_split[name] = _summarize(per_clip, corpus)
            logger.info(
                "[%s] %-16s wer_mean=%.4f wer_corpus=%.4f (n=%d)",
                s, name, per_split[name]["wer_mean"],
                per_split[name]["wer_corpus"], per_split[name]["n"],
            )
        per_split["weights"] = {
            "uniform": dict(zip(MODEL_NAMES, uniform.tolist())),
            "weighted": dict(zip(MODEL_NAMES, weighted.tolist())),
        }
        results[s] = per_split

    if save_json:
        out = Path(save_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w") as f:
            json.dump({
                "parquet_path": parquet_path,
                "train_ratio": train_ratio,
                "val_ratio": val_ratio,
                "seed": seed,
                "results": results,
            }, f, indent=2)
        logger.info("Saved ROVER results to %s", save_json)


if __name__ == "__main__":
    app()
