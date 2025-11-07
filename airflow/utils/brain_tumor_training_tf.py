#!/usr/bin/env python3
"""TensorFlow retraining utilities derived from the `brain_tumor_clf*.ipynb` notebooks."""

from __future__ import annotations

import json
import math
import os
import random
import shutil
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    jaccard_score,
    roc_auc_score,
)
from sklearn.preprocessing import label_binarize

from .brain_tumor_training import (
    DATASET_SUMMARY_FILE,
    ensure_data_dir,
    resolve_dataset_dir,
    resolve_output_dir,
    save_dataset_summary,
    summarize_dataset,
)

TF_METADATA_FILE = "tf_training_metadata.json"
TF_EVALUATION_FILE = "tf_evaluation_metrics.json"
TF_TEST_RESULTS_FILE = "tf_test_results.json"
TF_SPLIT_MANIFEST = "tf_split_manifest.json"
TF_MODEL_DIRNAME = "tf_model"
TF_CONFUSION_MATRIX = "tf_confusion_matrix.png"
TF_CLASSIFICATION_REPORT = "tf_classification_report.txt"
TF_TENSORBOARD_ROOT = "tf_logs"

DEFAULT_TRAIN_RATIO = 0.8
DEFAULT_VAL_RATIO = 0.1
DEFAULT_TEST_RATIO = 0.1
DEFAULT_IMG_SIZE = 224
DEFAULT_BATCH_SIZE = 32
DEFAULT_EPOCHS = 100
DEFAULT_PATIENCE = 10
DEFAULT_SEED = 42
DEFAULT_SHUFFLE_BUFFER = 1000


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def _class_dirs(dataset_dir: Path) -> List[Path]:
    return sorted([path for path in dataset_dir.iterdir() if path.is_dir()])


def _copy_file(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.symlink(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def prepare_tf_split(
    dataset_dir: Path,
    output_dir: Path,
    train_ratio: float = DEFAULT_TRAIN_RATIO,
    val_ratio: float = DEFAULT_VAL_RATIO,
    test_ratio: float = DEFAULT_TEST_RATIO,
    seed: int = DEFAULT_SEED,
) -> Dict[str, Any]:
    """Replicate the manual dataset splitting performed in the notebook."""

    dataset_dir = Path(dataset_dir)
    output_dir = ensure_data_dir(Path(output_dir))

    if not math.isclose(train_ratio + val_ratio + test_ratio, 1.0, rel_tol=1e-6):
        raise ValueError("Train/val/test ratios must sum to 1.0")

    class_dirs = _class_dirs(dataset_dir)
    if not class_dirs:
        raise ValueError(f"No class sub-directories found under {dataset_dir}")

    split_root = output_dir / "tf_split"
    if split_root.exists():
        shutil.rmtree(split_root)

    rng = random.Random(seed)
    class_names = [path.name for path in class_dirs]
    split_counts: Dict[str, Dict[str, int]] = {
        split: {name: 0 for name in class_names} for split in ("train", "val", "test")
    }

    for class_dir in class_dirs:
        files = [path for path in class_dir.iterdir() if path.is_file()]
        rng.shuffle(files)
        total = len(files)
        train_end = int(total * train_ratio)
        val_end = train_end + int(total * val_ratio)
        train_files = files[:train_end]
        val_files = files[train_end:val_end]
        test_files = files[val_end:]

        # handle rounding leftovers
        assigned = len(train_files) + len(val_files) + len(test_files)
        if assigned < total:
            test_files.extend(files[assigned:])

        for split_name, candidates in (
            ("train", train_files),
            ("val", val_files),
            ("test", test_files),
        ):
            for src in candidates:
                dst = split_root / split_name / class_dir.name / src.name
                _copy_file(src, dst)
            split_counts[split_name][class_dir.name] = len(candidates)

    manifest = {
        "dataset_dir": str(dataset_dir),
        "split_dir": str(split_root),
        "class_names": class_names,
        "train_ratio": train_ratio,
        "val_ratio": val_ratio,
        "test_ratio": test_ratio,
        "seed": seed,
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "split_counts": split_counts,
    }

    manifest_path = output_dir / TF_SPLIT_MANIFEST
    with open(manifest_path, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)

    aggregate_counts = {name: 0 for name in class_names}
    for class_name in class_names:
        aggregate_counts[class_name] = sum(split_counts[split][class_name] for split in ("train", "val", "test"))
    summary = {
        "split_dir": str(split_root),
        "class_names": class_names,
        "split_counts": split_counts,
        "aggregate_counts": aggregate_counts,
        "total_images": sum(aggregate_counts.values()),
    }
    save_dataset_summary(output_dir, summary)

    return manifest


def load_tf_split_manifest(output_dir: Path) -> Dict[str, Any]:
    manifest_path = Path(output_dir) / TF_SPLIT_MANIFEST
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"TensorFlow split manifest missing at {manifest_path}. Run prepare_tf_split first."
        )
    with open(manifest_path, "r", encoding="utf-8") as fh:
        return json.load(fh)


@dataclass
class TfDatasets:
    train: tf.data.Dataset
    val: tf.data.Dataset
    test: tf.data.Dataset
    class_names: List[str]


def build_tf_datasets(
    split_dir: Path,
    img_size: int = DEFAULT_IMG_SIZE,
    batch_size: int = DEFAULT_BATCH_SIZE,
    seed: int = DEFAULT_SEED,
    shuffle_buffer: int = DEFAULT_SHUFFLE_BUFFER,
) -> TfDatasets:
    split_dir = Path(split_dir)
    if not split_dir.exists():
        raise FileNotFoundError(f"Split directory not found at {split_dir}")

    common_kwargs = dict(
        image_size=(img_size, img_size),
        batch_size=batch_size,
        label_mode="int",
        interpolation="bilinear",
    )
    train_ds = tf.keras.utils.image_dataset_from_directory(
        split_dir / "train", shuffle=True, seed=seed, **common_kwargs
    )
    val_ds = tf.keras.utils.image_dataset_from_directory(
        split_dir / "val", shuffle=False, seed=seed, **common_kwargs
    )
    test_ds = tf.keras.utils.image_dataset_from_directory(
        split_dir / "test", shuffle=False, seed=seed, **common_kwargs
    )

    class_names = train_ds.class_names
    autotune = tf.data.AUTOTUNE
    train_ds = train_ds.cache().shuffle(shuffle_buffer).prefetch(autotune)
    val_ds = val_ds.cache().prefetch(autotune)
    test_ds = test_ds.cache().prefetch(autotune)

    return TfDatasets(train=train_ds, val=val_ds, test=test_ds, class_names=class_names)


def create_tf_classifier(num_classes: int, img_size: int = DEFAULT_IMG_SIZE) -> tf.keras.Model:
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Rescaling(1.0 / 255, input_shape=(img_size, img_size, 3)),
            tf.keras.layers.Conv2D(16, 3, padding="same", activation="relu"),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Conv2D(32, 3, padding="same", activation="relu"),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Conv2D(64, 3, padding="same", activation="relu"),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dense(num_classes),
        ],
        name="brain_tumor_cnn",
    )
    return model


def _collect_predictions(model: tf.keras.Model, dataset: tf.data.Dataset) -> tuple[np.ndarray, np.ndarray]:
    y_true_batches: List[np.ndarray] = []
    y_prob_batches: List[np.ndarray] = []

    for images, labels in dataset:
        logits = model(images, training=False)
        probs = tf.nn.softmax(logits, axis=1).numpy()
        y_true_batches.append(labels.numpy())
        y_prob_batches.append(probs)

    return np.concatenate(y_true_batches), np.concatenate(y_prob_batches)


def _metrics_from_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    class_names: Iterable[str],
) -> Dict[str, Any]:
    class_names = list(class_names)
    n_classes = len(class_names)

    acc = float(accuracy_score(y_true, y_pred))
    f1_macro = float(f1_score(y_true, y_pred, average="macro"))
    f1_per_class = {
        class_names[i]: float(score)
        for i, score in enumerate(f1_score(y_true, y_pred, average=None, labels=list(range(n_classes))))
    }

    cm = confusion_matrix(y_true, y_pred, labels=list(range(n_classes)))
    iou_per_class: Dict[str, float] = {}
    for idx, name in enumerate(class_names):
        tp = cm[idx, idx]
        fp = cm[:, idx].sum() - tp
        fn = cm[idx, :].sum() - tp
        denom = tp + fp + fn
        iou_per_class[name] = float(tp / denom) if denom else 0.0
    mean_iou = float(np.mean(list(iou_per_class.values())))

    if n_classes == 2:
        if y_prob.ndim == 1 or y_prob.shape[1] == 1:
            y_score = y_prob.reshape(-1)
        else:
            y_score = y_prob[:, 1]
        try:
            roc_auc = float(roc_auc_score(y_true, y_score))
        except ValueError:
            roc_auc = float("nan")
        roc_auc_per_class = {class_names[1]: roc_auc}
    else:
        y_true_bin = label_binarize(y_true, classes=list(range(n_classes)))
        try:
            roc_auc = float(
                roc_auc_score(y_true_bin, y_prob, average="macro", multi_class="ovr")
            )
            roc_auc_per_class = {}
            for idx, name in enumerate(class_names):
                try:
                    roc_auc_per_class[name] = float(roc_auc_score(y_true_bin[:, idx], y_prob[:, idx]))
                except ValueError:
                    roc_auc_per_class[name] = float("nan")
        except ValueError:
            roc_auc = float("nan")
            roc_auc_per_class = {name: float("nan") for name in class_names}

    return {
        "accuracy": acc,
        "f1_macro": f1_macro,
        "f1_per_class": f1_per_class,
        "iou_per_class": iou_per_class,
        "mean_iou": mean_iou,
        "roc_auc": roc_auc,
        "roc_auc_per_class": roc_auc_per_class,
        "confusion_matrix": cm.tolist(),
    }


def _save_confusion_matrix(path: Path, cm: np.ndarray, class_names: Iterable[str]) -> None:
    import seaborn as sns

    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix - Brain Tumor MRI (TF)")
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()


def _write_report(path: Path, report: str) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(report)


def _evaluate_model(
    model: tf.keras.Model,
    dataset: tf.data.Dataset,
    class_names: Iterable[str],
    artifact_dir: Path | None = None,
) -> Dict[str, Any]:
    y_true, y_prob = _collect_predictions(model, dataset)
    y_pred = np.argmax(y_prob, axis=1)
    metrics = _metrics_from_predictions(y_true, y_pred, y_prob, class_names)

    report = classification_report(y_true, y_pred, target_names=list(class_names), digits=4)
    metrics["classification_report"] = report

    if artifact_dir is not None:
        cm_array = np.array(metrics["confusion_matrix"])
        cm_path = Path(artifact_dir) / TF_CONFUSION_MATRIX
        _save_confusion_matrix(cm_path, cm_array, class_names)
        report_path = Path(artifact_dir) / TF_CLASSIFICATION_REPORT
        _write_report(report_path, report)
        metrics["confusion_matrix_path"] = str(cm_path)
        metrics["classification_report_path"] = str(report_path)

    return metrics


def train_tf_model_pipeline(
    data_dir: Path,
    output_dir: Path,
    batch_size: int = DEFAULT_BATCH_SIZE,
    epochs: int = DEFAULT_EPOCHS,
    img_size: int = DEFAULT_IMG_SIZE,
    patience: int = DEFAULT_PATIENCE,
    seed: int = DEFAULT_SEED,
) -> Dict[str, Any]:
    data_dir = Path(data_dir)
    output_dir = ensure_data_dir(Path(output_dir))

    manifest = load_tf_split_manifest(output_dir)
    split_dir = Path(manifest["split_dir"])

    _seed_everything(seed)

    datasets = build_tf_datasets(split_dir, img_size=img_size, batch_size=batch_size, seed=seed)
    class_names = datasets.class_names

    model = create_tf_classifier(len(class_names), img_size=img_size)
    model.compile(
        optimizer="adam",
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )

    log_root = ensure_data_dir(output_dir / TF_TENSORBOARD_ROOT)
    run_id = time.strftime("%Y%m%d-%H%M%S")
    callbacks = [
        tf.keras.callbacks.TensorBoard(log_dir=str(log_root / run_id), histogram_freq=1),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=patience, restore_best_weights=True
        ),
    ]

    start = time.time()
    history = model.fit(
        datasets.train,
        validation_data=datasets.val,
        epochs=epochs,
        callbacks=callbacks,
        verbose=2,
    )
    training_minutes = float((time.time() - start) / 60)

    metrics = _evaluate_model(model, datasets.test, class_names, artifact_dir=output_dir)

    model_dir = output_dir / TF_MODEL_DIRNAME
    if model_dir.exists():
        shutil.rmtree(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    saved_model_dir = model_dir / "saved_model"
    model.save(saved_model_dir)

    keras_path = model_dir / "brain_tumor_classifier.keras"
    model.save(keras_path)

    h5_path = model_dir / "brain_tumor_classifier.h5"
    model.save(h5_path)

    history_records: List[Dict[str, Any]] = []
    num_epochs = len(history.history.get("loss", []))
    for epoch_idx in range(num_epochs):
        history_records.append(
            {
                "epoch": epoch_idx + 1,
                "loss": float(history.history["loss"][epoch_idx]),
                "accuracy": float(history.history["accuracy"][epoch_idx]),
                "val_loss": float(history.history["val_loss"][epoch_idx]),
                "val_accuracy": float(history.history["val_accuracy"][epoch_idx]),
            }
        )

    metadata = {
        "data_dir": str(data_dir),
        "output_dir": str(output_dir),
        "split_dir": str(split_dir),
        "class_names": class_names,
        "hyperparameters": {
            "batch_size": batch_size,
            "epochs": epochs,
            "img_size": img_size,
            "patience": patience,
            "seed": seed,
        },
        "model_paths": {
            "saved_model_dir": str(saved_model_dir),
            "keras_path": str(keras_path),
            "h5_path": str(h5_path),
        },
        "metrics": metrics,
        "history": history_records,
        "tensorboard_log_dir": str(log_root / run_id),
        "training_minutes": training_minutes,
        "generated_at": datetime.utcnow().isoformat() + "Z",
    }

    metadata_path = output_dir / TF_METADATA_FILE
    with open(metadata_path, "w", encoding="utf-8") as fh:
        json.dump(metadata, fh, indent=2)

    return metadata


def load_tf_metadata(output_dir: Path) -> Dict[str, Any]:
    metadata_path = Path(output_dir) / TF_METADATA_FILE
    if not metadata_path.exists():
        raise FileNotFoundError(f"TensorFlow metadata missing at {metadata_path}")
    with open(metadata_path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def evaluate_tf_best_model(output_dir: Path) -> Dict[str, Any]:
    metadata = load_tf_metadata(output_dir)
    class_names = metadata["class_names"]
    split_dir = Path(metadata["split_dir"])
    hyperparams = metadata.get("hyperparameters", {})
    batch_size = int(hyperparams.get("batch_size", DEFAULT_BATCH_SIZE))
    img_size = int(hyperparams.get("img_size", DEFAULT_IMG_SIZE))

    datasets = build_tf_datasets(split_dir, img_size=img_size, batch_size=batch_size)

    saved_model_dir = Path(metadata["model_paths"]["saved_model_dir"])
    if not saved_model_dir.exists():
        raise FileNotFoundError(f"SavedModel directory missing at {saved_model_dir}")

    model = tf.keras.models.load_model(saved_model_dir)
    metrics = _evaluate_model(model, datasets.test, class_names, artifact_dir=Path(output_dir))

    evaluation_path = Path(output_dir) / TF_EVALUATION_FILE
    with open(evaluation_path, "w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2)

    return metrics


def run_tf_inference_samples(
    output_dir: Path,
    sample_count: int = 5,
    seed: int = DEFAULT_SEED,
) -> List[Dict[str, Any]]:
    metadata = load_tf_metadata(output_dir)
    split_dir = Path(metadata["split_dir"])
    class_names = metadata["class_names"]
    hyperparams = metadata.get("hyperparameters", {})
    img_size = int(hyperparams.get("img_size", DEFAULT_IMG_SIZE))

    dataset = tf.keras.utils.image_dataset_from_directory(
        split_dir / "test",
        image_size=(img_size, img_size),
        batch_size=1,
        shuffle=True,
        seed=seed,
    )

    saved_model_dir = Path(metadata["model_paths"]["saved_model_dir"])
    model = tf.keras.models.load_model(saved_model_dir)

    samples: List[Dict[str, Any]] = []

    for images, labels in dataset.unbatch().take(sample_count):
        logits = model(images[None, ...], training=False)
        probs = tf.nn.softmax(logits, axis=1).numpy().flatten()
        pred_idx = int(np.argmax(probs))
        label_idx = int(labels.numpy())
        samples.append(
            {
                "actual": class_names[label_idx],
                "predicted": class_names[pred_idx],
                "confidence": float(probs[pred_idx]),
                "probabilities": {class_names[i]: float(prob) for i, prob in enumerate(probs)},
            }
        )

    results_path = Path(output_dir) / TF_TEST_RESULTS_FILE
    with open(results_path, "w", encoding="utf-8") as fh:
        json.dump(samples, fh, indent=2)

    return samples
