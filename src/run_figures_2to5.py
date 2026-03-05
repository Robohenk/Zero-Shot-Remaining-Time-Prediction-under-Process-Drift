#!/usr/bin/env python3
"""
Figure-driven analysis for remaining-time prediction under drift.

Inputs:
- Event logs named ExpNRunM.txt (tab-separated; may have UTF-8 BOM).
- Optionally: baseline LSTM training + prediction generation.
- Or: load existing predictions from --pred_dir.

Outputs:
- Figures (PNG + PDF) + raw points (CSV; optionally full Parquet) per figure folder.
- Includes a dataset sanity figure: percent of cases whose first observed event is 'arrivalAtSource'.

Figures / analyses:
Fig1: Case-start coverage counter (arrivalAtSource rate + first-event distribution)
Fig2: Run-level variability + bootstrap CI (MAE/RMSE)
Fig3: Severity ladder across scenario groups (source-test vs target vs alt)
Fig4: Error anatomy (by remaining time, signed error, progress)
Fig5: Drift magnitude vs error (PSI/JS components) + correlation

Note:
- If you do NOT want to retrain, run with --pred_dir containing per-run prediction parquets.
- If you DO want to train baseline LSTM, run with --train_and_predict (requires tensorflow).
"""

from __future__ import annotations

import argparse
import dataclasses
import math
import os
import json
import re
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from scipy import stats
from tqdm import tqdm


# -------------------------
# Timing helpers (milestones)
# -------------------------

def fmt_seconds(s: float) -> str:
    s = float(s)
    if s < 60:
        return f"{s:.1f}s"
    m = int(s // 60)
    r = s - 60 * m
    if m < 60:
        return f"{m}m {r:.0f}s"
    h = int(m // 60)
    m2 = m - 60 * h
    return f"{h}h {m2}m"


@contextmanager
def stage(name: str):
    t0 = time.time()
    print(f"\n[STAGE] {name} ...", flush=True)
    yield
    dt = time.time() - t0
    print(f"[DONE ] {name} in {fmt_seconds(dt)}", flush=True)


# -------------------------
# TensorFlow - GPU visibility
# -------------------------
def tf_detect_gpus() -> int:
    """Number of GPUs visible to TensorFlow (CUDA GPUs in practice)."""
    try:
        import tensorflow as tf
        return len(tf.config.list_physical_devices("GPU"))
    except Exception:
        return 0


# -------------------------
# Configuration (edit here)
# -------------------------

SOURCE_EXPS = set(range(10, 19))   # 10..18
TARGET_EXPS = set(range(1, 10))    # 1..9
ALT_EXPS    = set(range(19, 28))   # 19..27

# In your paper protocol: 18 runs train / 2 runs test for each source experiment.
DEFAULT_SOURCE_TEST_RUNS = {19, 20}

FILENAME_RE = re.compile(r"Exp(?P<exp>\d+)Run(?P<run>\d+)\.txt$", re.IGNORECASE)


# -------------------------
# Utilities
# -------------------------

@dataclasses.dataclass(frozen=True)
class RunId:
    exp: int
    run: int

    @property
    def key(self) -> str:
        return f"Exp{self.exp}Run{self.run}"


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def save_manifest(out_dir: Path, name: str, payload: dict) -> None:
    ensure_dir(out_dir)
    with (out_dir / f"{name}_manifest.json").open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def bootstrap_ci(x: np.ndarray, n_boot: int = 2000, ci: float = 0.95, seed: int = 7) -> Tuple[float, float]:
    """Percentile bootstrap CI for the mean."""
    rng = np.random.default_rng(seed)
    x = x[~np.isnan(x)]
    if len(x) == 0:
        return (np.nan, np.nan)
    boots = rng.choice(x, size=(n_boot, len(x)), replace=True).mean(axis=1)
    alpha = (1 - ci) / 2
    return (float(np.quantile(boots, alpha)), float(np.quantile(boots, 1 - alpha)))


# -------------------------
# Loading + prefixing
# -------------------------

def discover_runs(data_root: Path) -> Dict[RunId, Path]:
    runs: Dict[RunId, Path] = {}
    for p in data_root.rglob("Exp*Run*.txt"):
        m = FILENAME_RE.search(p.name)
        if not m:
            continue
        rid = RunId(exp=int(m.group("exp")), run=int(m.group("run")))
        runs[rid] = p
    return dict(sorted(runs.items(), key=lambda kv: (kv[0].exp, kv[0].run)))


def load_log(path: Path) -> pd.DataFrame:
    """
    Loads one ExpNRunM log file.
    - Tab separated
    - Often has UTF-8 BOM, so use utf-8-sig.
    """
    df = pd.read_csv(path, sep="\t", encoding="utf-8-sig", low_memory=False)
    df.columns = [c.strip() for c in df.columns]
    # Guard against duplicate column labels in raw files
    if df.columns.duplicated().any():
        dups = df.columns[df.columns.duplicated()].tolist()
        print(f"[load_log] WARNING: duplicate columns in {path.name}: {dups} (keeping first)", flush=True)
        df = df.loc[:, ~df.columns.duplicated(keep="first")].copy()
    return df


def _parse_timestamp_series(ts: pd.Series) -> pd.Series:
    """
    Deterministic parsing with strict format first, then fallback.
    Expected format: YYYY/MM/DD HH:MM:SS.ffff
    """
    ts1 = pd.to_datetime(ts, format="%Y/%m/%d %H:%M:%S.%f", errors="coerce")
    if ts1.isna().any():
        ts2 = pd.to_datetime(ts, errors="coerce")
        ts1 = ts1.fillna(ts2)
    return ts1


def clean_and_sort_log(df: pd.DataFrame) -> pd.DataFrame:
    if "productIDStr" not in df.columns:
        raise ValueError("Expected column 'productIDStr' not found.")
    if "timeStamp" not in df.columns:
        raise ValueError("Expected column 'timeStamp' not found.")

    df = df.copy()

    # Keep only rows with a usable case id
    df["productIDStr"] = df["productIDStr"].astype(str)
    df = df[df["productIDStr"].notna()]
    df = df[df["productIDStr"].str.lower().ne("na")]
    df = df[df["productIDStr"].str.strip().ne("")]

    # Parse timestamps
    df["timeStamp"] = _parse_timestamp_series(df["timeStamp"])
    df = df[df["timeStamp"].notna()]

    # tie-breaking (same timestamp may repeat)
    if "uniqueID" in df.columns:
        df["uniqueID"] = pd.to_numeric(df["uniqueID"], errors="coerce")
    else:
        df["uniqueID"] = np.arange(len(df), dtype=float)

    df = df.sort_values(["productIDStr", "timeStamp", "uniqueID"], kind="mergesort")

    # Fill missing categories
    for c in ["event", "vehicleType", "vehicle", "processingStation"]:
        if c in df.columns:
            df[c] = df[c].fillna("NA").astype(str)
        else:
            df[c] = "NA"

    if "currentDecayLevel" in df.columns:
        df["currentDecayLevel"] = pd.to_numeric(df["currentDecayLevel"], errors="coerce")
    else:
        df["currentDecayLevel"] = np.nan

    return df


def build_prefix_table(df: pd.DataFrame, rid: RunId) -> pd.DataFrame:
    """
    Builds a prefix-level table with:
    - case_id, prefix_len, case_len, prefix_time
    - y_true: remaining time in seconds (case_end_time - prefix_time)
    - plus per-prefix summary features (for drift metrics / anatomy plots)
    """
    df = clean_and_sort_log(df)

    # Case boundaries + lengths
    end_times = df.groupby("productIDStr")["timeStamp"].max().rename("case_end_time")
    start_times = df.groupby("productIDStr")["timeStamp"].min().rename("case_start_time")
    case_lens = df.groupby("productIDStr").size().rename("case_len")

    df = df.join(end_times, on="productIDStr")
    df = df.join(start_times, on="productIDStr")
    df = df.join(case_lens, on="productIDStr")

    # Time since previous event within case
    df["prev_time"] = df.groupby("productIDStr")["timeStamp"].shift(1)
    df["delta_t"] = (df["timeStamp"] - df["prev_time"]).dt.total_seconds()
    df.loc[df["delta_t"].isna(), "delta_t"] = 0.0

    # Position in trace (1..case_len)
    df["prefix_len"] = df.groupby("productIDStr").cumcount() + 1

    # Remaining time label
    df["y_true"] = (df["case_end_time"] - df["timeStamp"]).dt.total_seconds().astype(float)
    df.loc[df["y_true"] < 0, "y_true"] = 0.0

    # Prefix progress
    df["progress"] = df["prefix_len"] / df["case_len"].clip(lower=1)

    prefix_tbl = df[[
        "productIDStr", "event", "vehicleType", "vehicle", "processingStation",
        "currentDecayLevel", "timeStamp", "case_start_time", "case_end_time",
        "case_len", "prefix_len", "progress", "delta_t", "y_true"
    ]].copy()

    prefix_tbl.insert(0, "run", rid.run)
    prefix_tbl.insert(0, "exp", rid.exp)
    prefix_tbl.insert(0, "run_id", rid.key)

    return prefix_tbl


# -------------------------
# Dataset sanity figure: case-start counter
# -------------------------

def label_group(exp: int, run: int, source_test_runs: set[int]) -> str:
    if exp in TARGET_EXPS:
        return "target (Exp1–9)"
    if exp in SOURCE_EXPS:
        if run in source_test_runs:
            return "source-test (Exp10–18)"
        return "source-train (Exp10–18)"
    if exp in ALT_EXPS:
        return "alt (Exp19–27)"
    return "other"


def fig1_case_start_counter(
    runs: Dict[RunId, Path],
    out_root: Path,
    source_test_runs: set[int],
    start_event_name: str = "arrivalAtSource",
    pred: Optional[pd.DataFrame] = None,
) -> None:
    """
    Produces:
    - per-run %cases whose first observed event == start_event_name
    - distribution of first observed events per group
    Uses predictions if they contain event+prefix_len; otherwise loads logs.
    """
    out_dir = out_root / "fig1"
    ensure_dir(out_dir)

    with stage("Fig1: compute first-event coverage"):
        if pred is not None and {"run_id", "productIDStr", "prefix_len", "event", "exp", "run"}.issubset(set(pred.columns)):
            first_rows = pred[pred["prefix_len"] == 1].copy()
            first_rows = first_rows.sort_values(["run_id", "productIDStr"]).drop_duplicates(["run_id", "productIDStr"])
            first_rows["group"] = [label_group(e, r, source_test_runs) for e, r in zip(first_rows["exp"], first_rows["run"])]
            src = "predictions"
        else:
            rows = []
            for rid, path in tqdm(runs.items(), desc="[fig1] scanning logs"):
                df = load_log(path)
                df = clean_and_sort_log(df)
                first = (
                    df.sort_values(["productIDStr", "timeStamp", "uniqueID"], kind="mergesort")
                      .groupby("productIDStr", as_index=False)
                      .first()[["productIDStr", "event"]]
                )
                first["run_id"] = rid.key
                first["exp"] = rid.exp
                first["run"] = rid.run
                rows.append(first)
            first_rows = pd.concat(rows, ignore_index=True)
            first_rows["group"] = [label_group(e, r, source_test_runs) for e, r in zip(first_rows["exp"], first_rows["run"])]
            src = "logs"

        per_run = (
            first_rows.groupby(["run_id", "exp", "run", "group"], as_index=False)
                     .agg(
                         n_cases=("productIDStr", "nunique"),
                         n_start=("event", lambda s: int((s == start_event_name).sum())),
                     )
        )
        per_run["start_event_rate"] = per_run["n_start"] / per_run["n_cases"].clip(lower=1)

        grp_rows = []
        for g, gdf in per_run.groupby("group"):
            rates = gdf["start_event_rate"].values
            lo, hi = bootstrap_ci(rates)
            grp_rows.append({
                "group": g,
                "n_runs": len(gdf),
                "rate_mean": float(np.mean(rates)) if len(rates) else np.nan,
                "rate_ci_lo": lo,
                "rate_ci_hi": hi,
            })
        per_group = pd.DataFrame(grp_rows).sort_values("group")

        dist = (
            first_rows.groupby(["group", "event"], as_index=False)
                      .agg(n=("productIDStr", "size"))
        )
        dist["share"] = dist.groupby("group")["n"].transform(lambda x: x / x.sum())

    # Export
    per_run.to_csv(out_dir / "fig1_case_start_rate_by_run.csv", index=False)
    per_group.to_csv(out_dir / "fig1_case_start_rate_by_group.csv", index=False)
    dist.to_csv(out_dir / "fig1_first_event_distribution.csv", index=False)

    # Plot
    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(8.5, 4), constrained_layout=True)

    order = ["source-test (Exp10–18)", "target (Exp1–9)", "alt (Exp19–27)", "source-train (Exp10–18)"]
    per_group2 = per_group.set_index("group").reindex([g for g in order if g in per_group["group"].values]).reset_index()

    x = np.arange(len(per_group2))
    ax.bar(x, per_group2["rate_mean"], color="#4C72B0")
    ax.errorbar(
        x, per_group2["rate_mean"],
        yerr=[per_group2["rate_mean"] - per_group2["rate_ci_lo"], per_group2["rate_ci_hi"] - per_group2["rate_mean"]],
        fmt="none", ecolor="black", capsize=4, lw=1
    )
    ax.set_xticks(x)
    ax.set_xticklabels(per_group2["group"], rotation=15, ha="right")
    ax.set_ylim(0, 1)
    ax.set_ylabel(f"Fraction of cases with first event = '{start_event_name}'")
    ax.set_title("Fig1 — Case-start coverage (first observed event)")

    fig.savefig(out_dir / "fig1_case_start_counter.png", dpi=250, bbox_inches="tight")
    fig.savefig(out_dir / "fig1_case_start_counter.pdf", bbox_inches="tight")

    save_manifest(out_dir, "fig1", {
        "what": "Counter for cases starting at arrivalAtSource (sanity check for mid-flow starts)",
        "computed_from": src,
        "start_event_name": start_event_name,
        "exports": [
            "fig1_case_start_rate_by_run.csv",
            "fig1_case_start_rate_by_group.csv",
            "fig1_first_event_distribution.csv",
            "fig1_case_start_counter.png/pdf",
        ]
    })


# -------------------------
# Optional model: simple baseline LSTM
# -------------------------

def try_import_tf() -> bool:
    try:
        import tensorflow as tf  # noqa
        _ = tf.__version__
        return True
    except Exception:
        return False


@dataclasses.dataclass
class Encoding:
    event_to_idx: Dict[str, int]
    veh_to_idx: Dict[str, int]
    max_len: int
    num_mean: np.ndarray
    num_std: np.ndarray
    # IMPORTANT: default so inference workers can instantiate without it
    n_train_prefix_rows: int = 0


def combine_mean_m2(n1: int, mean1: np.ndarray, m2_1: np.ndarray,
                    n2: int, mean2: np.ndarray, m2_2: np.ndarray) -> Tuple[int, np.ndarray, np.ndarray]:
    """Combine (n, mean, M2) stats elementwise for vectors."""
    if n1 == 0:
        return n2, mean2, m2_2
    if n2 == 0:
        return n1, mean1, m2_1
    delta = mean2 - mean1
    n = n1 + n2
    mean = mean1 + delta * (n2 / n)
    m2 = m2_1 + m2_2 + (delta ** 2) * (n1 * n2 / n)
    return n, mean, m2


def fit_encoding_stream(
    train_ids: List[RunId],
    runs: Dict[RunId, Path],
    max_len_cap: int = 200,
) -> Encoding:
    """
    Streaming fit:
    - collects event/vehicle categories from training runs
    - estimates numeric mean/std for features: [delta_t, time_since_start, decay, prefix_pos_norm]
    - computes max case length capped at max_len_cap
    """
    events = set()
    vehs = set()
    max_len = 0
    n_train_prefix_rows = 0

    n = 0
    mean = np.zeros(4, dtype=float)
    m2 = np.zeros(4, dtype=float)

    for rid in tqdm(train_ids, desc="[encoding] scanning training runs"):
        df = load_log(runs[rid])
        pt = build_prefix_table(df, rid)
        n_train_prefix_rows += len(pt)

        events.update(pt["event"].astype(str).unique())
        vehs.update(pt["vehicleType"].astype(str).unique())
        max_len = max(max_len, int(pt["case_len"].max()))

        time_since_start = (pt["timeStamp"] - pt["case_start_time"]).dt.total_seconds().astype(float).values
        delta_t = pt["delta_t"].astype(float).values

        decay = pt["currentDecayLevel"].astype(float)
        if decay.isna().all():
            decay = decay.fillna(0.0)
        else:
            decay = decay.fillna(decay.median())
        decay = decay.values

        prefix_pos_norm = (pt["prefix_len"] / pt["case_len"].clip(lower=1)).astype(float).values

        num = np.vstack([delta_t, time_since_start, decay, prefix_pos_norm]).T
        num = num[~np.isnan(num).any(axis=1)]
        n2 = len(num)
        if n2 > 0:
            mean2 = num.mean(axis=0)
            var2 = num.var(axis=0, ddof=0)
            m2_2 = var2 * n2
            n, mean, m2 = combine_mean_m2(n, mean, m2, n2, mean2, m2_2)

    max_len = min(max_len, max_len_cap)
    std = np.sqrt(m2 / max(n, 1))
    std[std == 0] = 1.0

    event_list = ["<UNK>"] + sorted(events)
    veh_list = ["<UNK>"] + sorted(vehs)

    return Encoding(
        event_to_idx={e: i for i, e in enumerate(event_list)},
        veh_to_idx={v: i for i, v in enumerate(veh_list)},
        max_len=max_len,
        num_mean=mean,
        num_std=std,
        n_train_prefix_rows=int(n_train_prefix_rows),
    )


def case_to_event_rows(prefix_tbl: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    return {cid: g.sort_values("prefix_len") for cid, g in prefix_tbl.groupby("productIDStr")}


def make_prefix_samples_for_case(case_df: pd.DataFrame, enc: Encoding) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """
    For one case, create:
    - X: (n_prefixes, max_len, feat_dim)
    - y: (n_prefixes,)
    - meta: DataFrame with prefix metadata, INCLUDING event/vehicleType/delta_t/decay for drift analyses.
    """
    events = case_df["event"].astype(str).values
    vehs = case_df["vehicleType"].astype(str).values

    time_since_start = (case_df["timeStamp"] - case_df["case_start_time"]).dt.total_seconds().astype(float).values
    delta_t = case_df["delta_t"].astype(float).values

    decay = case_df["currentDecayLevel"].astype(float)
    if decay.isna().all():
        decay = decay.fillna(0.0)
    else:
        decay = decay.fillna(decay.median())
    decay = decay.values

    prefix_pos_norm = (case_df["prefix_len"] / case_df["case_len"].clip(lower=1)).astype(float).values

    num = np.vstack([delta_t, time_since_start, decay, prefix_pos_norm]).T
    num = (num - enc.num_mean) / enc.num_std

    E = len(enc.event_to_idx)
    V = len(enc.veh_to_idx)
    feat_dim = 4 + E + V

    case_len = len(case_df)
    max_len = enc.max_len

    n_prefixes = case_len
    X = np.zeros((n_prefixes, max_len, feat_dim), dtype=np.float32)
    y = case_df["y_true"].astype(float).values

    ev_idx = np.array([enc.event_to_idx.get(e, 0) for e in events], dtype=int)
    vh_idx = np.array([enc.veh_to_idx.get(v, 0) for v in vehs], dtype=int)

    ev_oh = np.zeros((case_len, E), dtype=np.float32)
    vh_oh = np.zeros((case_len, V), dtype=np.float32)
    ev_oh[np.arange(case_len), ev_idx] = 1.0
    vh_oh[np.arange(case_len), vh_idx] = 1.0

    base = np.hstack([num, ev_oh, vh_oh]).astype(np.float32)

    for k in range(1, case_len + 1):
        seq = base[:k]
        if k > max_len:
            seq = seq[-max_len:]
        start = max_len - len(seq)
        X[k - 1, start:, :] = seq

    meta = case_df[[
        "exp", "run", "run_id", "productIDStr", "case_len",
        "prefix_len", "progress", "timeStamp",
        "event", "vehicleType",
        "delta_t", "currentDecayLevel",
        "y_true"
    ]].copy()

    return X, y, meta


def build_model(enc: Encoding):
    import tensorflow as tf

    feat_dim = 4 + len(enc.event_to_idx) + len(enc.veh_to_idx)

    inp = tf.keras.Input(shape=(enc.max_len, feat_dim), name="x")
    x = tf.keras.layers.Masking(mask_value=0.0)(inp)
    x = tf.keras.layers.LSTM(50, return_sequences=False)(x)
    out = tf.keras.layers.Dense(1, activation="linear")(x)

    model = tf.keras.Model(inputs=inp, outputs=out)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="mae",
        metrics=[
            tf.keras.metrics.MeanAbsoluteError(name="mae"),
            tf.keras.metrics.RootMeanSquaredError(name="rmse"),
        ],
    )
    return model


# -------------------------
# CPU-parallel prediction
# -------------------------

def _predict_worker_cpu(run_ids, runs, out_dir, model_path, enc_path, batch_size):
    # Force CPU
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

    import tensorflow as tf
    import json
    import numpy as np
    import pandas as pd
    from tqdm import tqdm

    model = tf.keras.models.load_model(model_path)

    with open(enc_path, "r", encoding="utf-8") as f:
        encj = json.load(f)

    enc = Encoding(
        event_to_idx=encj["event_to_idx"],
        veh_to_idx=encj["veh_to_idx"],
        max_len=int(encj["max_len"]),
        num_mean=np.array(encj["num_mean"], dtype=float),
        num_std=np.array(encj["num_std"], dtype=float),
    )

    ensure_dir(out_dir)

    for rid in tqdm(run_ids, desc=f"[cpu-worker] runs", leave=False):
        df = load_log(runs[rid])
        pt = build_prefix_table(df, rid)
        cases = case_to_event_rows(pt)

        metas, preds = [], []
        for _, cdf in cases.items():
            X, y_true, meta = make_prefix_samples_for_case(cdf, enc)
            y_pred = model.predict(X, batch_size=batch_size, verbose=0).reshape(-1)
            
            # meta already contains y_true, so store only y_pred here (avoid duplicate y_true columns)
            preds.append(pd.DataFrame({"y_pred": y_pred}))
            metas.append(meta.reset_index(drop=True))

        meta_all = pd.concat(metas, ignore_index=True) if metas else pd.DataFrame()
        pred_all = pd.concat(preds, ignore_index=True) if preds else pd.DataFrame()

        out = pd.concat([meta_all, pred_all], axis=1)
        
        # Safety: drop any accidental duplicate columns
        out = out.loc[:, ~pd.Index(out.columns).duplicated(keep="first")]
        
        out["abs_err"] = (out["y_pred"] - out["y_true"]).abs()
        out["signed_err"] = out["y_pred"] - out["y_true"]
        out.to_parquet(out_dir / f"{rid.key}.parquet", index=False)


def parallel_predict_on_cpu(eval_ids, runs, out_dir, model_path, enc_path, batch_size, cpu_workers: int):
    from concurrent.futures import ProcessPoolExecutor

    chunks = [[] for _ in range(cpu_workers)]
    for i, rid in enumerate(eval_ids):
        chunks[i % cpu_workers].append(rid)

    ensure_dir(out_dir)

    print(f"[predict] CPU-parallel: {len(eval_ids)} runs over {cpu_workers} workers "
          f"(~{math.ceil(len(eval_ids)/max(cpu_workers,1))} runs/worker)", flush=True)

    with ProcessPoolExecutor(max_workers=cpu_workers) as ex:
        futs = []
        for chunk in chunks:
            if chunk:
                futs.append(ex.submit(_predict_worker_cpu, chunk, runs, out_dir, model_path, enc_path, batch_size))
        for f in tqdm(futs, desc="[predict] workers done"):
            f.result()


# -------------------------
# Training + prediction
# -------------------------

def train_and_predict(
    runs: Dict[RunId, Path],
    data_root: Optional[Path],  # kept for compatibility with your call; not used
    out_root: Path,
    source_test_runs: set[int],
    epochs: int = 1,
    batch_size: int = 128,
    max_len_cap: int = 200,
    args=None,
) -> Path:
    """
    Trains on source exps (10-18) excluding source_test_runs, then predicts on:
    - source test runs (10-18, test runs only)
    - target runs (1-9, all)
    - alt runs (19-27, all)

    Writes per-run predictions parquet to out_root/predictions/.
    Returns predictions_dir.
    """
    if not try_import_tf():
        raise RuntimeError(
            "TensorFlow not installed. Either install tensorflow, or run with --pred_dir to analyze existing predictions."
        )

    import tensorflow as tf  # noqa

    predictions_dir = out_root / "predictions"
    ensure_dir(predictions_dir)

    train_ids = [rid for rid in runs.keys()
                 if rid.exp in SOURCE_EXPS and rid.run not in source_test_runs]
    test_ids_source = [rid for rid in runs.keys()
                       if rid.exp in SOURCE_EXPS and rid.run in source_test_runs]
    eval_ids = [rid for rid in runs.keys()
                if (rid.exp in TARGET_EXPS) or (rid.exp in ALT_EXPS)] + test_ids_source
    eval_ids = sorted(set(eval_ids), key=lambda r: (r.exp, r.run))

    with stage(f"Fit encodings (streaming) on {len(train_ids)} training runs"):
        enc = fit_encoding_stream(train_ids, runs, max_len_cap=max_len_cap)

    # Training dataset generator
    def gen_train():
        for rid in train_ids:
            df = load_log(runs[rid])
            pt = build_prefix_table(df, rid)
            cases = case_to_event_rows(pt)
            for _, cdf in cases.items():
                X, y, _ = make_prefix_samples_for_case(cdf, enc)
                for i in range(len(y)):
                    yield X[i], y[i]

    feat_dim = 4 + len(enc.event_to_idx) + len(enc.veh_to_idx)

    ds_train = tf.data.Dataset.from_generator(
        gen_train,
        output_signature=(
            tf.TensorSpec(shape=(enc.max_len, feat_dim), dtype=tf.float32),
            tf.TensorSpec(shape=(), dtype=tf.float32),
        ),
    )

    steps_per_epoch = int(np.ceil(enc.n_train_prefix_rows / max(batch_size, 1)))
    ds_train = ds_train.shuffle(20000).repeat().batch(batch_size).prefetch(tf.data.AUTOTUNE)

    model = build_model(enc)
    model.summary()

    with stage(f"Train baseline LSTM (epochs={epochs}, steps/epoch={steps_per_epoch})"):
        model.fit(ds_train, epochs=epochs, steps_per_epoch=steps_per_epoch, verbose=1)

    # SAVE artifacts BEFORE prediction (needed for CPU-parallel workers)
    model_path = out_root / "baseline_lstm.keras"
    enc_path = out_root / "encoding.json"
    with stage("Save baseline artifacts (model + encoding)"):
        model.save(model_path)
        with enc_path.open("w", encoding="utf-8") as f:
            json.dump({
                "event_to_idx": enc.event_to_idx,
                "veh_to_idx": enc.veh_to_idx,
                "max_len": enc.max_len,
                "num_mean": enc.num_mean.tolist(),
                "num_std": enc.num_std.tolist(),
            }, f, indent=2)

    with stage(f"Predict on evaluation runs ({len(eval_ids)})"):
        gpu_count = tf_detect_gpus()

        if args is not None and args.parallel_predict_cpu and gpu_count == 0:
            print(f"[predict] TF sees 0 GPUs -> using CPU-parallel prediction with {args.cpu_workers} workers.", flush=True)
            parallel_predict_on_cpu(
                eval_ids=eval_ids,
                runs=runs,
                out_dir=predictions_dir,
                model_path=model_path,
                enc_path=enc_path,
                batch_size=batch_size,
                cpu_workers=args.cpu_workers,
            )
        else:
            print("[predict] Using single-process prediction.", flush=True)
            for i, rid in enumerate(tqdm(eval_ids, desc="[predict] runs")):
                df = load_log(runs[rid])
                pt = build_prefix_table(df, rid)
                cases = case_to_event_rows(pt)

                metas = []
                preds = []
                for _, cdf in cases.items():
                    X, y_true, meta = make_prefix_samples_for_case(cdf, enc)
                    y_pred = model.predict(X, batch_size=batch_size, verbose=0).reshape(-1)
                    preds.append(pd.DataFrame({"y_pred": y_pred}))
                    metas.append(meta.reset_index(drop=True))

                meta_all = pd.concat(metas, ignore_index=True) if metas else pd.DataFrame()
                pred_all = pd.concat(preds, ignore_index=True) if preds else pd.DataFrame()

                out = pd.concat([meta_all, pred_all], axis=1)
                out = out.loc[:, ~pd.Index(out.columns).duplicated(keep="first")]
                
                out["abs_err"] = (out["y_pred"] - out["y_true"]).abs()
                out["signed_err"] = out["y_pred"] - out["y_true"]

                out_path = predictions_dir / f"{rid.key}.parquet"
                out.to_parquet(out_path, index=False)

                # milestone print every 10 runs
                if (i + 1) % 10 == 0 or (i + 1) == len(eval_ids):
                    print(f"[predict] saved {i+1}/{len(eval_ids)} -> {out_path.name} (rows={len(out)})", flush=True)

    return predictions_dir


# -------------------------
# Predictions IO
# -------------------------

def load_predictions_dir(pred_dir: Path) -> pd.DataFrame:
    parts = []
    for p in sorted(pred_dir.glob("Exp*Run*.parquet")):
        parts.append(pd.read_parquet(p))
    if not parts:
        raise FileNotFoundError(f"No parquet files found in {pred_dir}")
    df = pd.concat(parts, ignore_index=True)

    if df.columns.duplicated().any():
        dups = df.columns[df.columns.duplicated()].tolist()
        print(f"[load_predictions_dir] WARNING: duplicate columns: {dups} (keeping first)", flush=True)
        df = df.loc[:, ~df.columns.duplicated(keep="first")].copy()

    if "abs_err" not in df.columns:
        df["abs_err"] = (df["y_pred"] - df["y_true"]).abs()
    if "signed_err" not in df.columns:
        df["signed_err"] = (df["y_pred"] - df["y_true"])

    return df


# -------------------------
# FIG 2: Run-level variability + CI (item 2)
# -------------------------

def fig2_run_variability(pred: pd.DataFrame, out_root: Path, source_test_runs: set[int]) -> None:
    out_dir = out_root / "fig2"
    ensure_dir(out_dir)

    df = pred.copy()
    df["group"] = [label_group(e, r, source_test_runs) for e, r in zip(df["exp"], df["run"])]

    run_metrics = (
        df.groupby(["exp", "run", "run_id", "group"], as_index=False)
          .agg(MAE=("abs_err", "mean"),
               RMSE=("signed_err", lambda x: float(np.sqrt(np.mean(np.square(x))))))
    )

    eval_rm = run_metrics[run_metrics["group"].isin(
        ["source-test (Exp10–18)", "target (Exp1–9)", "alt (Exp19–27)"]
    )].copy()

    rows = []
    for g, gdf in eval_rm.groupby("group"):
        mae = gdf["MAE"].values
        rmse = gdf["RMSE"].values
        mae_ci = bootstrap_ci(mae)
        rmse_ci = bootstrap_ci(rmse)
        rows.append({
            "group": g,
            "n_runs": len(gdf),
            "MAE_mean": float(np.mean(mae)) if len(mae) else np.nan,
            "MAE_ci_lo": mae_ci[0],
            "MAE_ci_hi": mae_ci[1],
            "RMSE_mean": float(np.mean(rmse)) if len(rmse) else np.nan,
            "RMSE_ci_lo": rmse_ci[0],
            "RMSE_ci_hi": rmse_ci[1],
        })
    group_summary = pd.DataFrame(rows).sort_values("group")

    eval_rm.to_csv(out_dir / "fig2_run_metrics.csv", index=False)
    group_summary.to_csv(out_dir / "fig2_group_summary.csv", index=False)

    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)

    sns.violinplot(data=eval_rm, x="group", y="MAE", inner="box", ax=axes[0])
    axes[0].set_title("Fig2a — Run-level MAE variability")
    axes[0].set_xlabel("")
    axes[0].set_ylabel("MAE (s)")
    axes[0].tick_params(axis="x", rotation=20)

    sns.violinplot(data=eval_rm, x="group", y="RMSE", inner="box", ax=axes[1])
    axes[1].set_title("Fig2b — Run-level RMSE variability")
    axes[1].set_xlabel("")
    axes[1].set_ylabel("RMSE (s)")
    axes[1].tick_params(axis="x", rotation=20)

    fig.savefig(out_dir / "fig2_run_variability.png", dpi=250)
    fig.savefig(out_dir / "fig2_run_variability.pdf")

    save_manifest(out_dir, "fig2", {
        "what": "Run-level variability + bootstrap CI (item 2)",
        "exports": ["fig2_run_metrics.csv", "fig2_group_summary.csv", "fig2_run_variability.png/pdf"]
    })


# -------------------------
# FIG 3: Severity ladder across scenario groups (item 3)
# -------------------------

def fig3_severity_ladder(pred: pd.DataFrame, out_root: Path, source_test_runs: set[int]) -> None:
    out_dir = out_root / "fig3"
    ensure_dir(out_dir)

    df = pred.copy()
    df["group"] = [label_group(e, r, source_test_runs) for e, r in zip(df["exp"], df["run"])]

    run_metrics = (
        df.groupby(["exp", "run", "run_id", "group"], as_index=False)
          .agg(MAE=("abs_err", "mean"),
               RMSE=("signed_err", lambda x: float(np.sqrt(np.mean(np.square(x))))))
    )
    eval_rm = run_metrics[run_metrics["group"].isin(
        ["source-test (Exp10–18)", "target (Exp1–9)", "alt (Exp19–27)"]
    )].copy()

    rows = []
    for g, gdf in eval_rm.groupby("group"):
        mae = gdf["MAE"].values
        rmse = gdf["RMSE"].values
        mae_ci = bootstrap_ci(mae)
        rmse_ci = bootstrap_ci(rmse)
        rows.append({
            "group": g,
            "n_runs": len(gdf),
            "MAE_mean": float(np.mean(mae)),
            "MAE_ci_lo": mae_ci[0], "MAE_ci_hi": mae_ci[1],
            "RMSE_mean": float(np.mean(rmse)),
            "RMSE_ci_lo": rmse_ci[0], "RMSE_ci_hi": rmse_ci[1],
        })
    group_table = pd.DataFrame(rows).sort_values("group")

    eval_rm.to_csv(out_dir / "fig3_run_metrics.csv", index=False)
    group_table.to_csv(out_dir / "fig3_group_table.csv", index=False)

    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(8.5, 4), constrained_layout=True)

    order = ["source-test (Exp10–18)", "target (Exp1–9)", "alt (Exp19–27)"]
    gt = group_table.set_index("group").loc[[g for g in order if g in group_table["group"].values]].reset_index()

    x = np.arange(len(gt))
    ax.bar(x - 0.18, gt["MAE_mean"], width=0.35, label="MAE", color="#4C72B0")
    ax.bar(x + 0.18, gt["RMSE_mean"], width=0.35, label="RMSE", color="#55A868")

    ax.errorbar(x - 0.18, gt["MAE_mean"],
                yerr=[gt["MAE_mean"] - gt["MAE_ci_lo"], gt["MAE_ci_hi"] - gt["MAE_mean"]],
                fmt="none", ecolor="black", capsize=4, lw=1)
    ax.errorbar(x + 0.18, gt["RMSE_mean"],
                yerr=[gt["RMSE_mean"] - gt["RMSE_ci_lo"], gt["RMSE_ci_hi"] - gt["RMSE_mean"]],
                fmt="none", ecolor="black", capsize=4, lw=1)

    ax.set_xticks(x)
    ax.set_xticklabels(gt["group"], rotation=15, ha="right")
    ax.set_ylabel("Error (s)")
    ax.set_title("Fig3 — Performance across scenario groups (severity ladder)")
    ax.legend()

    fig.savefig(out_dir / "fig3_severity_ladder.png", dpi=250)
    fig.savefig(out_dir / "fig3_severity_ladder.pdf")

    save_manifest(out_dir, "fig3", {
        "what": "Performance across scenario groups (item 3)",
        "exports": ["fig3_run_metrics.csv", "fig3_group_table.csv", "fig3_severity_ladder.png/pdf"]
    })


# -------------------------
# FIG 4: Error anatomy (item 4)
# -------------------------

def fig4_error_anatomy(
    pred: pd.DataFrame,
    out_root: Path,
    source_test_runs: set[int],
    n_bins_rt: int = 12,
    n_bins_prog: int = 10,
    plot_sample: int = 200_000,
    export_full_points: bool = False,
) -> None:
    out_dir = out_root / "fig4"
    ensure_dir(out_dir)

    df = pred.copy()
    df["group"] = [label_group(e, r, source_test_runs) for e, r in zip(df["exp"], df["run"])]
    df = df[df["group"].isin(["source-test (Exp10–18)", "target (Exp1–9)", "alt (Exp19–27)"])].copy()

    df_plot = df
    if len(df) > plot_sample:
        df_plot = df.sample(plot_sample, random_state=7)

    if export_full_points:
        df.to_parquet(out_dir / "fig4_full_points.parquet", index=False)
    df_plot.to_csv(out_dir / "fig4_points_used_for_plots.csv", index=False)

    df["rt_bin"] = pd.qcut(df["y_true"], q=n_bins_rt, duplicates="drop")
    binned_rt = (
        df.groupby(["group", "rt_bin"], as_index=False)
          .agg(y_true_mid=("y_true", "median"),
               MAE=("abs_err", "mean"),
               n=("abs_err", "size"))
    )
    ci_rows = []
    for (g, b), sub in df.groupby(["group", "rt_bin"]):
        lo, hi = bootstrap_ci(sub["abs_err"].values)
        ci_rows.append({"group": g, "rt_bin": b, "MAE_ci_lo": lo, "MAE_ci_hi": hi})
    binned_rt = binned_rt.merge(pd.DataFrame(ci_rows), on=["group", "rt_bin"], how="left")
    binned_rt.to_csv(out_dir / "fig4a_mae_by_remaining_time_bin.csv", index=False)

    df["rt_bin2"] = pd.qcut(df["y_true"], q=n_bins_rt, duplicates="drop")
    binned_signed = (
        df.groupby(["group", "rt_bin2"], as_index=False)
          .agg(y_true_mid=("y_true", "median"),
               signed_err_mean=("signed_err", "mean"),
               signed_err_median=("signed_err", "median"),
               n=("signed_err", "size"))
    )
    binned_signed.to_csv(out_dir / "fig4b_signed_error_by_remaining_time_bin.csv", index=False)

    df["prog_bin"] = pd.cut(df["progress"].clip(0, 1), bins=np.linspace(0, 1, n_bins_prog + 1), include_lowest=True)
    binned_prog = (
        df.groupby(["group", "prog_bin"], as_index=False)
          .agg(progress_mid=("progress", "median"),
               MAE=("abs_err", "mean"),
               n=("abs_err", "size"))
    )
    ci_rows2 = []
    for (g, b), sub in df.groupby(["group", "prog_bin"]):
        lo, hi = bootstrap_ci(sub["abs_err"].values)
        ci_rows2.append({"group": g, "prog_bin": b, "MAE_ci_lo": lo, "MAE_ci_hi": hi})
    binned_prog = binned_prog.merge(pd.DataFrame(ci_rows2), on=["group", "prog_bin"], how="left")
    binned_prog.to_csv(out_dir / "fig4c_mae_by_progress_bin.csv", index=False)

    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8), constrained_layout=True)

    for g, sub in binned_rt.groupby("group"):
        axes[0].plot(sub["y_true_mid"], sub["MAE"], marker="o", label=g)
        axes[0].fill_between(sub["y_true_mid"], sub["MAE_ci_lo"], sub["MAE_ci_hi"], alpha=0.15)
    axes[0].set_title("Fig4a — MAE vs remaining time")
    axes[0].set_xlabel("True remaining time (s)")
    axes[0].set_ylabel("MAE (s)")

    for g, sub in binned_signed.groupby("group"):
        axes[1].plot(sub["y_true_mid"], sub["signed_err_mean"], marker="o", label=g)
    axes[1].axhline(0, color="black", lw=1)
    axes[1].set_title("Fig4b — Signed error vs remaining time")
    axes[1].set_xlabel("True remaining time (s)")
    axes[1].set_ylabel("Mean signed error (s)")

    for g, sub in binned_prog.groupby("group"):
        axes[2].plot(sub["progress_mid"], sub["MAE"], marker="o", label=g)
        axes[2].fill_between(sub["progress_mid"], sub["MAE_ci_lo"], sub["MAE_ci_hi"], alpha=0.15)
    axes[2].set_title("Fig4c — MAE vs prefix progress")
    axes[2].set_xlabel("Progress (prefix_len / case_len)")
    axes[2].set_ylabel("MAE (s)")

    handles, labels = axes[2].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.02))

    fig.savefig(out_dir / "fig4_error_anatomy.png", dpi=250, bbox_inches="tight")
    fig.savefig(out_dir / "fig4_error_anatomy.pdf", bbox_inches="tight")

    save_manifest(out_dir, "fig4", {
        "what": "Error anatomy (item 4)",
        "exports": [
            "fig4_points_used_for_plots.csv",
            "fig4a_mae_by_remaining_time_bin.csv",
            "fig4b_signed_error_by_remaining_time_bin.csv",
            "fig4c_mae_by_progress_bin.csv",
            "fig4_error_anatomy.png/pdf",
            "fig4_full_points.parquet (optional)"
        ],
        "plot_sample": plot_sample,
        "export_full_points": export_full_points
    })


# -------------------------
# FIG 5: Drift magnitude vs error (item 5)
# -------------------------

def psi_numeric(train: np.ndarray, test: np.ndarray, n_bins: int = 10) -> float:
    """Population Stability Index for numeric feature."""
    train = train[~np.isnan(train)]
    test = test[~np.isnan(test)]
    if len(train) < 50 or len(test) < 50:
        return np.nan
    qs = np.quantile(train, np.linspace(0, 1, n_bins + 1))
    qs = np.unique(qs)
    if len(qs) < 3:
        return np.nan
    train_counts, _ = np.histogram(train, bins=qs)
    test_counts, _ = np.histogram(test, bins=qs)
    train_p = train_counts / max(train_counts.sum(), 1)
    test_p = test_counts / max(test_counts.sum(), 1)
    eps = 1e-6
    train_p = np.clip(train_p, eps, 1)
    test_p = np.clip(test_p, eps, 1)
    return float(np.sum((test_p - train_p) * np.log(test_p / train_p)))


def js_divergence_categorical(train_vals: pd.Series, test_vals: pd.Series) -> float:
    """Jensen–Shannon divergence for categorical distributions."""
    train_counts = train_vals.value_counts(normalize=True)
    test_counts = test_vals.value_counts(normalize=True)
    cats = sorted(set(train_counts.index).union(set(test_counts.index)))
    p = np.array([train_counts.get(c, 0.0) for c in cats], dtype=float)
    q = np.array([test_counts.get(c, 0.0) for c in cats], dtype=float)
    m = 0.5 * (p + q)

    def kl(a, b):
        eps = 1e-12
        a = np.clip(a, eps, 1)
        b = np.clip(b, eps, 1)
        return np.sum(a * np.log(a / b))

    return float(0.5 * kl(p, m) + 0.5 * kl(q, m))


def fig5_drift_vs_error(
    pred: pd.DataFrame,
    out_root: Path,
    source_test_runs: set[int],
    runs: Dict[RunId, Path],
    ref_from_logs: bool = False,
    ref_sample_per_run: int = 50_000,
) -> None:
    out_dir = out_root / "fig5"
    ensure_dir(out_dir)

    df = pred.copy()
    df["group"] = [label_group(e, r, source_test_runs) for e, r in zip(df["exp"], df["run"])]

    train_ref = df[df["group"] == "source-train (Exp10–18)"].copy()
    
    # If we don't have source-train predictions, build reference from logs instead.
    if train_ref.empty and ref_from_logs:
        train_ids = [rid for rid in runs.keys()
                     if (rid.exp in SOURCE_EXPS) and (rid.run not in source_test_runs)]
        ref_parts = []
        for rid in tqdm(train_ids, desc="[fig5] build reference from logs"):
            pt = build_prefix_table(load_log(runs[rid]), rid)
    
            # keep only what Fig5 needs
            keep = pt[["y_true", "progress", "delta_t", "currentDecayLevel", "event", "vehicleType"]].copy()
    
            # sample to keep memory bounded
            n = min(ref_sample_per_run, len(keep))
            if n > 0:
                keep = keep.sample(n=n, random_state=7)
                ref_parts.append(keep)
    
        train_ref = pd.concat(ref_parts, ignore_index=True) if ref_parts else pd.DataFrame()
    
    # final fallback: use source-test if present
    if train_ref.empty:
        train_ref = df[df["group"] == "source-test (Exp10–18)"].copy()
    
    if train_ref.empty:
        raise RuntimeError("Fig5 needs a reference distribution. Provide source-test predictions or use --fig5_ref_from_logs.")


    run_err = (
        df[df["group"].isin(["source-test (Exp10–18)", "target (Exp1–9)", "alt (Exp19–27)"])]
        .groupby(["exp", "run", "run_id", "group"], as_index=False)
        .agg(MAE=("abs_err", "mean"))
    )

    drift_rows = []
    for (run_id, exp, run, grp), sub in tqdm(df.groupby(["run_id", "exp", "run", "group"]),
                                            desc="[fig5] compute drift scores"):
        if grp == "source-train (Exp10–18)":
            continue

        psi_y = psi_numeric(train_ref["y_true"].values, sub["y_true"].values)
        psi_p = psi_numeric(train_ref["progress"].values, sub["progress"].values)

        psi_dt = psi_numeric(train_ref["delta_t"].values, sub["delta_t"].values) if "delta_t" in sub.columns else np.nan
        psi_decay = psi_numeric(train_ref["currentDecayLevel"].values, sub["currentDecayLevel"].values) if "currentDecayLevel" in sub.columns else np.nan

        js_event = np.nan
        js_veh = np.nan
        if {"event", "vehicleType"}.issubset(set(sub.columns)) and {"event", "vehicleType"}.issubset(set(train_ref.columns)):
            js_event = js_divergence_categorical(train_ref["event"].astype(str), sub["event"].astype(str))
            js_veh = js_divergence_categorical(train_ref["vehicleType"].astype(str), sub["vehicleType"].astype(str))

        drift_rows.append({
            "run_id": run_id, "exp": exp, "run": run, "group": grp,
            "psi_y_true": psi_y,
            "psi_progress": psi_p,
            "psi_delta_t": psi_dt,
            "psi_decay": psi_decay,
            "js_event": js_event,
            "js_vehicleType": js_veh,
        })

    drift = pd.DataFrame(drift_rows)
    components = ["psi_y_true", "psi_progress", "psi_delta_t", "psi_decay", "js_event", "js_vehicleType"]
    drift["drift_score"] = drift[components].mean(axis=1, skipna=True)

    merged = drift.merge(run_err, on=["run_id", "exp", "run", "group"], how="inner")
    valid = merged.dropna(subset=["drift_score", "MAE"])

    spearman_r = np.nan
    if len(valid) >= 3:
        res = stats.spearmanr(valid["drift_score"], valid["MAE"])
        spearman_r = getattr(res, "correlation", getattr(res, "statistic", np.nan))

    drift.to_csv(out_dir / "fig5_drift_scores.csv", index=False)
    merged.to_csv(out_dir / "fig5_scatter_points.csv", index=False)

    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(7.5, 4), constrained_layout=True)

    sns.scatterplot(data=merged, x="drift_score", y="MAE", hue="group", ax=ax)
    if len(valid) >= 3:
        sns.regplot(data=valid, x="drift_score", y="MAE", scatter=False, ax=ax, color="black", line_kws={"lw": 1})

    ax.set_title(f"Fig5 — Drift magnitude vs MAE (Spearman r={spearman_r:.2f})" if not np.isnan(spearman_r)
                 else "Fig5 — Drift magnitude vs MAE")
    ax.set_xlabel("Drift score (avg of PSI/JS components)")
    ax.set_ylabel("MAE (s)")

    fig.savefig(out_dir / "fig5_drift_vs_error.png", dpi=250)
    fig.savefig(out_dir / "fig5_drift_vs_error.pdf")

    save_manifest(out_dir, "fig5", {
        "what": "Drift magnitude sanity check + relation to error (item 5)",
        "components": components,
        "spearman_r": None if np.isnan(spearman_r) else float(spearman_r),
        "exports": ["fig5_drift_scores.csv", "fig5_scatter_points.csv", "fig5_drift_vs_error.png/pdf"]
    })


# -------------------------
# Main
# -------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", type=str, required=True, help="Folder containing ExpNRunM.txt logs (recursively searched).")
    ap.add_argument("--out_root", type=str, default="outputs_2to5", help="Output folder for figures + exported points.")
    ap.add_argument("--figures", type=str, default="1,2,3,4,5", help="Comma-separated: 1,2,3,4,5 or 'all'")
    ap.add_argument("--source_test_runs", type=str, default="19,20", help="Comma-separated run numbers used as source test.")
    ap.add_argument("--train_and_predict", action="store_true",
                    help="If set: trains baseline LSTM on source-train and generates predictions parquets.")
    ap.add_argument("--pred_dir", type=str, default="",
                    help="If set: load existing per-run predictions parquets from here (skips training).")
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--max_len_cap", type=int, default=200)
    ap.add_argument("--plot_sample", type=int, default=200_000)
    ap.add_argument("--export_full_points", action="store_true")
    ap.add_argument("--start_event_name", type=str, default="arrivalAtSource",
                    help="For Fig1: count cases whose first observed event equals this string.")
    ap.add_argument("--parallel_predict_cpu", action="store_true",
                    help="Parallelize prediction over runs on CPU (recommended when TF sees no GPU).")
    ap.add_argument("--cpu_workers", type=int, default=max((os.cpu_count() or 4) // 2, 1),
                    help="Number of CPU worker processes for --parallel_predict_cpu.")
    ap.add_argument("--fig5_ref_from_logs", action="store_true",
                help="For Fig5: build reference distribution from source-train logs (Exp10–18 runs 1–18).")
    ap.add_argument("--fig5_ref_sample_per_run", type=int, default=50_000,
                help="Rows sampled per source-train run to build the Fig5 reference (keeps memory bounded).")


    args = ap.parse_args()

    gpu_count = tf_detect_gpus()
    if gpu_count == 0:
        print("[WARN] TensorFlow sees NO GPU (tf.config.list_physical_devices('GPU') == []).", flush=True)
        print("       On native Windows, TF CUDA GPU support is not available for TF >= 2.11.", flush=True)
        print("       Use --parallel_predict_cpu for speed, or run in WSL2 to use the NVIDIA GPU.", flush=True)

    data_root = Path(args.data_root)
    out_root = Path(args.out_root)
    ensure_dir(out_root)

    source_test_runs = {int(x) for x in args.source_test_runs.split(",") if x.strip()}

    if args.figures.strip().lower() == "all":
        figset = {1, 2, 3, 4, 5}
    else:
        figset = {int(x) for x in args.figures.split(",") if x.strip()}

    with stage("Discover runs"):
        runs = discover_runs(data_root)
        if not runs:
            raise FileNotFoundError(f"No ExpNRunM.txt files found under {data_root}")
        print(f"[info] discovered {len(runs)} runs under {data_root}", flush=True)

    pred = None
    if args.train_and_predict:
        with stage("Train + predict (baseline LSTM)"):
            pred_dir = train_and_predict(
                runs=runs,
                data_root=data_root,
                out_root=out_root,
                source_test_runs=source_test_runs,
                epochs=args.epochs,
                batch_size=args.batch_size,
                max_len_cap=args.max_len_cap,
                args=args,
            )
        with stage("Load predictions"):
            pred = load_predictions_dir(pred_dir)
    else:
        if args.pred_dir:
            with stage("Load predictions"):
                pred = load_predictions_dir(Path(args.pred_dir))

    # Fig1 can run even without predictions (it falls back to scanning logs).
    if 1 in figset:
        with stage("Generate Fig1"):
            fig1_case_start_counter(
                runs=runs,
                out_root=out_root,
                source_test_runs=source_test_runs,
                start_event_name=args.start_event_name,
                pred=pred,
            )

    if any(k in figset for k in [2, 3, 4, 5]) and pred is None:
        raise ValueError("Figures 2–5 require predictions. Provide --pred_dir or run with --train_and_predict.")

    if pred is not None:
        if 2 in figset:
            with stage("Generate Fig2"):
                fig2_run_variability(pred, out_root, source_test_runs)
        if 3 in figset:
            with stage("Generate Fig3"):
                fig3_severity_ladder(pred, out_root, source_test_runs)
        if 4 in figset:
            with stage("Generate Fig4"):
                fig4_error_anatomy(pred, out_root, source_test_runs,
                                   plot_sample=args.plot_sample,
                                   export_full_points=args.export_full_points)
        if 5 in figset:
            with stage("Generate Fig5"):
                fig5_drift_vs_error(
                    pred, out_root, source_test_runs,
                    runs=runs,
                    ref_from_logs=args.fig5_ref_from_logs,
                    ref_sample_per_run=args.fig5_ref_sample_per_run
                )


    print(f"\n[OK] Done. Outputs in: {out_root.resolve()}", flush=True)


if __name__ == "__main__":
    main()
