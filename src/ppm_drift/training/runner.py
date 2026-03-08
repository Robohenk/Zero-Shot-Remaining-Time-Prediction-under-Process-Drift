from __future__ import annotations

import csv
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import BenchmarkConfig
from ..constants import ALT_EXPS, SOURCE_EXPS, TARGET_EXPS, RunId
from ..data.loading import load_log
from ..data.preprocess import build_prefix_table
from ..evaluation.metrics import run_metrics_from_predictions
from ..features.encoding import Encoding, fit_encoding_stream
from ..features.sequence_builder import case_to_event_rows, make_prefix_samples_for_case, prefix_generator
from ..models import build_model
from ..utils.io import ensure_dir, dump_json
from ..utils.logging_utils import append_timing, update_status
from ..utils.repro import set_global_seed


def split_run_ids(runs: dict[RunId, Path], source_test_runs: set[int]):
    train_ids = [rid for rid in runs if rid.exp in SOURCE_EXPS and rid.run not in source_test_runs]
    source_test_ids = [rid for rid in runs if rid.exp in SOURCE_EXPS and rid.run in source_test_runs]
    target_ids = [rid for rid in runs if rid.exp in TARGET_EXPS]
    alt_ids = [rid for rid in runs if rid.exp in ALT_EXPS]
    eval_ids = sorted(source_test_ids + target_ids + alt_ids, key=lambda r: (r.exp, r.run))
    return train_ids, source_test_ids, target_ids, alt_ids, eval_ids


def _train_val_split(train_ids: list[RunId], validation_fraction: float, seed: int):
    rng = np.random.default_rng(seed)
    ordered = list(train_ids)
    rng.shuffle(ordered)
    n_val = max(1, int(round(len(ordered) * validation_fraction)))
    val_ids = ordered[:n_val]
    fit_ids = ordered[n_val:]
    return fit_ids, val_ids


def _tf_dataset(ids, runs, enc, batch_size, repeat, shuffle, build_prefix_table_fn=build_prefix_table):
    import tensorflow as tf

    ds = tf.data.Dataset.from_generator(
        lambda: prefix_generator(ids, runs, enc, load_log, build_prefix_table_fn),
        output_signature=(
            tf.TensorSpec(shape=(enc.max_len, enc.feature_dim), dtype=tf.float32),
            tf.TensorSpec(shape=(), dtype=tf.float32),
        ),
    )
    if shuffle:
        ds = ds.shuffle(20000, reshuffle_each_iteration=True)
    if repeat:
        ds = ds.repeat()
    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)


def _callbacks(output_dir: Path, cfg: BenchmarkConfig):
    import tensorflow as tf

    callbacks = [tf.keras.callbacks.CSVLogger(output_dir / "history.csv")]
    if cfg.train.use_early_stopping:
        callbacks.append(tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=cfg.train.patience, restore_best_weights=True))
    callbacks.append(tf.keras.callbacks.ModelCheckpoint(output_dir / "best_model.keras", monitor="val_loss", save_best_only=True))
    return callbacks


def train_one_model(runs: dict[RunId, Path], cfg: BenchmarkConfig, logger=None, work_root: str | Path | None = None) -> Path:
    import time
    set_global_seed(cfg.train.seed)
    work_root = Path(work_root or cfg.data.output_root)
    output_root = ensure_dir(Path(cfg.data.output_root))
    model_root = ensure_dir(output_root / cfg.train.model_name)

    train_ids, source_test_ids, target_ids, alt_ids, eval_ids = split_run_ids(runs, cfg.split.source_test_runs)
    fit_ids, val_ids = _train_val_split(train_ids, cfg.train.validation_fraction, cfg.train.seed)

    t0 = time.perf_counter()
    if logger: logger.info("Fitting encoding for %d training runs", len(fit_ids))
    update_status(work_root, "fit_encoding_stream", "running", {"current_model": cfg.train.model_name})
    enc = fit_encoding_stream(fit_ids, runs, max_len_cap=cfg.data.max_len_cap)
    append_timing(work_root / "logs", f"{cfg.train.model_name}_fit_encoding_stream", time.perf_counter() - t0, {"model": cfg.train.model_name})
    enc.to_json(model_root / "encoding.json")
    dump_json(model_root / "config_snapshot.json", {"data": asdict(cfg.data), "split": {"source_test_runs": sorted(cfg.split.source_test_runs)}, "train": asdict(cfg.train)})

    train_ds = _tf_dataset(fit_ids, runs, enc, cfg.train.batch_size, repeat=True, shuffle=True)
    val_ds = _tf_dataset(val_ids, runs, enc, cfg.train.batch_size, repeat=False, shuffle=False)

    train_kwargs = asdict(cfg.train).copy()
    train_kwargs.pop("model_name", None)
    model_wrapper = build_model(cfg.train.model_name, enc, **train_kwargs)
    model = model_wrapper.build()
    model_wrapper.model = model
    dump_json(model_root / 'model_summary.json', {
        'model_name': cfg.train.model_name,
        'seed': cfg.train.seed,
        'epochs': cfg.train.epochs,
        'batch_size': cfg.train.batch_size,
        'feature_dim': enc.feature_dim,
        'max_len': enc.max_len,
        'n_train_runs': len(train_ids),
        'n_fit_runs': len(fit_ids),
        'n_val_runs': len(val_ids),
        'n_eval_runs': len(eval_ids),
        'parameter_count': int(model.count_params()),
    })

    steps_per_epoch = int(np.ceil(enc.n_train_prefix_rows / max(len(train_ids), 1) / max(cfg.train.batch_size, 1))) * max(len(fit_ids), 1)
    t1 = time.perf_counter()
    if logger: logger.info("Training %s for %d epochs", cfg.train.model_name, cfg.train.epochs)
    update_status(work_root, "model_fit", "running", {"current_model": cfg.train.model_name, "epochs": cfg.train.epochs})
    history = model_wrapper.fit(
        train_ds,
        val_ds,
        epochs=cfg.train.epochs,
        steps_per_epoch=max(steps_per_epoch, 1),
        callbacks=_callbacks(model_root, cfg),
        verbose=1,
    )
    append_timing(work_root / "logs", f"{cfg.train.model_name}_model_fit", time.perf_counter() - t1, {"model": cfg.train.model_name, "epochs": cfg.train.epochs})
    model_wrapper.save(model_root / "final_model.keras")

    pd.DataFrame(history.history).to_csv(model_root / "history_summary.csv", index=False)

    pred_dir = ensure_dir(model_root / "predictions")
    t2 = time.perf_counter()
    if logger: logger.info("Predicting %d evaluation runs for %s", len(eval_ids), cfg.train.model_name)
    update_status(work_root, "predict_eval_runs", "running", {"current_model": cfg.train.model_name, "n_eval_runs": len(eval_ids)})
    for i, rid in enumerate(eval_ids, start=1):
        pt = build_prefix_table(load_log(runs[rid]), rid)
        metas = []
        preds = []
        for _, cdf in case_to_event_rows(pt).items():
            x, _, meta = make_prefix_samples_for_case(cdf, enc)
            y_pred = model_wrapper.predict(x, batch_size=cfg.train.batch_size, verbose=0).reshape(-1)
            part = meta.reset_index(drop=True)
            part["y_pred"] = y_pred
            part["abs_err"] = (part["y_pred"] - part["y_true"]).abs()
            part["signed_err"] = part["y_pred"] - part["y_true"]
            metas.append(part)
        out = pd.concat(metas, ignore_index=True) if metas else pd.DataFrame()
        out.to_parquet(pred_dir / f"{rid.key}.parquet", index=False)
        if logger and (i % 10 == 0 or i == len(eval_ids)):
            logger.info("Predicted %d/%d evaluation runs for %s", i, len(eval_ids), cfg.train.model_name)

    append_timing(work_root / "logs", f"{cfg.train.model_name}_predict_eval_runs", time.perf_counter() - t2, {"model": cfg.train.model_name, "n_eval_runs": len(eval_ids)})
    metrics = run_metrics_from_predictions(pred_dir, cfg.split.source_test_runs)
    metrics.to_csv(model_root / "run_metrics.csv", index=False)
    return model_root


def run_epoch_grid(runs: dict[RunId, Path], base_cfg: BenchmarkConfig, epochs: list[int], seeds: list[int], logger=None, work_root: str | Path | None = None) -> pd.DataFrame:
    rows = []
    for epoch in epochs:
        for seed in seeds:
            cfg = BenchmarkConfig(data=base_cfg.data, split=base_cfg.split, train=base_cfg.train)
            cfg.train.epochs = epoch
            cfg.train.seed = seed
            cfg.data.output_root = str(Path(base_cfg.data.output_root) / f"epoch_grid_{base_cfg.train.model_name}" / f"epochs_{epoch}_seed_{seed}")
            if logger: logger.info("Epoch-grid run: model=%s epochs=%s seed=%s", cfg.train.model_name, epoch, seed)
            model_root = train_one_model(runs, cfg, logger=logger, work_root=work_root or base_cfg.data.output_root)
            run_metrics = pd.read_csv(model_root / "run_metrics.csv")
            summary = run_metrics.groupby("group", as_index=False).agg(mae_mean=("mae", "mean"), rmse_mean=("rmse", "mean"))
            summary["model_name"] = cfg.train.model_name
            summary["epochs"] = epoch
            summary["seed"] = seed
            rows.append(summary)
    out = pd.concat(rows, ignore_index=True)
    out_path = Path(base_cfg.data.output_root) / f"epoch_grid_{base_cfg.train.model_name}" / "epoch_grid_summary.csv"
    ensure_dir(out_path.parent)
    out.to_csv(out_path, index=False)
    return out
