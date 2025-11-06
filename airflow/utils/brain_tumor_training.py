#!/usr/bin/env python3
"""Utility helpers for the Retrain Brain Tumor Model pipeline.

The functions in this module are adapted from the standalone training
scripts located under `train_model/` and provide reusable primitives for
Airflow tasks including data preparation, model training, evaluation, and
analysis/report generation.
"""

from __future__ import annotations

import json
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import matplotlib
matplotlib.use("Agg")  # Ensure headless environments can render figures
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image, UnidentifiedImageError
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    jaccard_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, models, transforms

try:  # TensorBoard is optional but recommended
    from torch.utils.tensorboard import SummaryWriter
except ImportError:  # pragma: no cover - handled gracefully at runtime
    SummaryWriter = None  # type: ignore

DEFAULT_IMG_SIZE = 224
DEFAULT_TEST_SIZE = 0.1
DEFAULT_VAL_RATIO = 1 / 9
DEFAULT_BATCH_SIZE = 32
DEFAULT_EPOCHS = 50
DEFAULT_LR = 1e-4
DEFAULT_PATIENCE = 5
DEFAULT_NUM_WORKERS = 4
DEFAULT_SEED = 42
BEST_MODEL_NAME = "best_model.pth"
METADATA_FILE = "training_metadata.json"
DATASET_SUMMARY_FILE = "dataset_summary.json"
EVALUATION_FILE = "evaluation_metrics.json"
TEST_RESULTS_FILE = "test_results.json"


def get_device() -> torch.device:
    """Return the best available device for training/inference."""

    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed: int = DEFAULT_SEED) -> None:
    """Seed Python, NumPy, and PyTorch RNGs for reproducibility."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_transforms(img_size: int = DEFAULT_IMG_SIZE) -> Tuple[transforms.Compose, transforms.Compose]:
    """Create training and validation transforms matching the reference script."""

    mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    train_tf = transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )
    val_tf = transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )
    return train_tf, val_tf


def split_dataset(dataset: datasets.ImageFolder, test_size: float = DEFAULT_TEST_SIZE, val_ratio: float = DEFAULT_VAL_RATIO) -> Tuple[List[int], List[int], List[int]]:
    """Split dataset indices into train/validation/test folds with stratification."""

    targets = dataset.targets
    indices = list(range(len(targets)))
    train_val_idx, test_idx = train_test_split(
        indices,
        test_size=test_size,
        stratify=targets,
        random_state=DEFAULT_SEED,
    )
    train_val_targets = [targets[i] for i in train_val_idx]
    train_idx, val_idx = train_test_split(
        train_val_idx,
        test_size=val_ratio,
        stratify=train_val_targets,
        random_state=DEFAULT_SEED,
    )
    return train_idx, val_idx, test_idx


def create_model(num_classes: int, dropout: float = 0.5, pretrained: bool = True) -> nn.Module:
    """Load a ResNet18 backbone and adapt the classification head."""

    model = models.resnet18(pretrained=pretrained)
    in_features = model.fc.in_features
    model.fc = nn.Sequential(nn.Dropout(dropout), nn.Linear(in_features, num_classes))
    return model


@dataclass
class DataLoaders:
    train: DataLoader
    val: DataLoader
    test: DataLoader
    class_names: List[str]


def prepare_dataloaders(
    data_dir: Path,
    batch_size: int = DEFAULT_BATCH_SIZE,
    num_workers: int = DEFAULT_NUM_WORKERS,
    img_size: int = DEFAULT_IMG_SIZE,
    test_size: float = DEFAULT_TEST_SIZE,
    val_ratio: float = DEFAULT_VAL_RATIO,
) -> DataLoaders:
    """Create train/validation/test loaders using the reference transforms."""

    train_tf, val_tf = build_transforms(img_size)
    dataset_train_like = datasets.ImageFolder(data_dir, transform=train_tf)
    dataset_val_like = datasets.ImageFolder(data_dir, transform=val_tf)
    class_names = dataset_train_like.classes

    train_idx, val_idx, test_idx = split_dataset(dataset_train_like, test_size=test_size, val_ratio=val_ratio)

    train_ds = Subset(dataset_train_like, train_idx)
    val_ds = Subset(dataset_val_like, val_idx)
    test_ds = Subset(dataset_val_like, test_idx)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return DataLoaders(train=train_loader, val=val_loader, test=test_loader, class_names=class_names)


@torch.no_grad()
def evaluate_model(model: nn.Module, loader: DataLoader, device: torch.device) -> Tuple[float, float, np.ndarray, np.ndarray, np.ndarray]:
    """Evaluate a model returning loss, accuracy, probabilities, predictions, and labels."""

    model.eval()
    criterion = nn.CrossEntropyLoss()
    total_loss, total_correct = 0.0, 0
    all_probs: List[np.ndarray] = []
    all_preds: List[np.ndarray] = []
    all_labels: List[np.ndarray] = []

    for inputs, labels in loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        total_loss += loss.item() * inputs.size(0)
        probs = torch.softmax(outputs, dim=1)
        preds = probs.argmax(1)
        total_correct += torch.sum(preds == labels).item()
        all_probs.append(probs.cpu().numpy())
        all_preds.append(preds.cpu().numpy())
        all_labels.append(labels.cpu().numpy())

    avg_loss = total_loss / len(loader.dataset)
    avg_acc = total_correct / len(loader.dataset)
    return (
        avg_loss,
        avg_acc,
        np.concatenate(all_probs),
        np.concatenate(all_preds),
        np.concatenate(all_labels),
    )


def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _init_tensorboard(log_dir: Path) -> Any:
    if SummaryWriter is None:
        return None
    run_name = time.strftime("model_%Y-%m-%d_%H-%M")
    writer = SummaryWriter(log_dir=str(log_dir / run_name))
    return writer


def _close_tensorboard(writer: Any) -> None:
    if writer is not None:
        writer.close()


def train_model_pipeline(
    data_dir: Path,
    output_dir: Path,
    batch_size: int = DEFAULT_BATCH_SIZE,
    epochs: int = DEFAULT_EPOCHS,
    lr: float = DEFAULT_LR,
    patience: int = DEFAULT_PATIENCE,
    num_workers: int = DEFAULT_NUM_WORKERS,
    img_size: int = DEFAULT_IMG_SIZE,
    seed: int = DEFAULT_SEED,
    use_tensorboard: bool = True,
) -> Dict[str, Any]:
    """Train the brain tumor classifier and persist best checkpoint/metrics."""

    set_seed(seed)
    device = get_device()
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)

    model_dir = _ensure_dir(output_dir / "model")
    log_dir = _ensure_dir(output_dir / "log")

    loaders = prepare_dataloaders(data_dir, batch_size, num_workers, img_size)
    model = create_model(len(loaders.class_names)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)

    writer = _init_tensorboard(log_dir) if use_tensorboard else None

    best_val_acc = 0.0
    best_model_path = model_dir / BEST_MODEL_NAME
    no_improve = 0
    history: List[Dict[str, float]] = []

    start_time = time.time()
    for epoch in range(epochs):
        model.train()
        epoch_start = time.time()
        train_loss = 0.0
        train_correct = 0
        for inputs, labels in loaders.train:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * inputs.size(0)
            train_correct += torch.sum(outputs.argmax(1) == labels).item()

        train_loss /= len(loaders.train.dataset)  # type: ignore[arg-type]
        train_acc = train_correct / len(loaders.train.dataset)  # type: ignore[arg-type]
        val_loss, val_acc, _, _, _ = evaluate_model(model, loaders.val, device)

        epoch_time = time.time() - epoch_start
        history.append(
            {
                "epoch": epoch + 1,
                "train_loss": float(train_loss),
                "train_acc": float(train_acc),
                "val_loss": float(val_loss),
                "val_acc": float(val_acc),
                "epoch_minutes": float(epoch_time / 60),
            }
        )

        print(
            f"Epoch {epoch + 1}/{epochs} | Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} | ⏱ {epoch_time / 60:.2f} min"
        )

        if writer is not None:
            writer.add_scalar("Loss/train", train_loss, epoch + 1)
            writer.add_scalar("Loss/val", val_loss, epoch + 1)
            writer.add_scalar("Accuracy/train", train_acc, epoch + 1)
            writer.add_scalar("Accuracy/val", val_acc, epoch + 1)
            writer.add_scalar("Learning Rate", lr, epoch + 1)
            writer.add_scalar("Time/epoch_minutes", epoch_time / 60, epoch + 1)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            no_improve = 0
            torch.save(model.state_dict(), best_model_path)
        else:
            no_improve += 1
            if no_improve >= patience:
                print("⏹ Early stopping triggered.")
                break

    total_minutes = (time.time() - start_time) / 60
    print(f"Training complete in {total_minutes:.2f} minutes")
    _close_tensorboard(writer)

    # Evaluate best checkpoint on the test split
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    test_loss, test_acc, probs, preds, labels = evaluate_model(model, loaders.test, device)
    f1 = f1_score(labels, preds, average="weighted")
    iou = jaccard_score(labels, preds, average="weighted", labels=np.unique(preds))
    try:
        roc_auc = roc_auc_score(labels, probs, multi_class="ovr", average="weighted")
    except Exception:  # pragma: no cover - fallback when roc fails
        roc_auc = float("nan")

    metrics = {
        "device": str(device),
        "epochs_completed": len(history),
        "best_val_acc": float(best_val_acc),
        "test_loss": float(test_loss),
        "test_acc": float(test_acc),
        "f1_weighted": float(f1),
        "iou_weighted": float(iou),
        "roc_auc_weighted": float(roc_auc),
        "training_minutes": float(total_minutes),
    }

    # Persist artifacts
    confusion_path = log_dir / "confusion_matrix.png"
    _save_confusion_matrix(confusion_path, labels, preds, loaders.class_names)
    report_path = log_dir / "classification_report.txt"
    _write_text_report(report_path, labels, preds, loaders.class_names)

    metadata = {
        "data_dir": str(data_dir),
        "output_dir": str(output_dir),
        "model_dir": str(model_dir),
        "log_dir": str(log_dir),
        "class_names": loaders.class_names,
        "hyperparameters": {
            "batch_size": batch_size,
            "epochs": epochs,
            "lr": lr,
            "patience": patience,
            "num_workers": num_workers,
            "img_size": img_size,
            "seed": seed,
        },
        "best_model_path": str(best_model_path),
        "metrics": metrics,
        "history": history,
        "confusion_matrix_path": str(confusion_path),
        "classification_report_path": str(report_path),
    }

    with open(output_dir / METADATA_FILE, "w", encoding="utf-8") as fp:
        json.dump(metadata, fp, indent=2)

    print(f"✅ Test Accuracy: {test_acc:.4f}")
    print(
        f"F1: {f1:.4f} | IoU: {iou:.4f} | ROC-AUC: {roc_auc:.4f}\n"
        f"Artifacts saved to {output_dir}"
    )

    return metadata


def _save_confusion_matrix(path: Path, labels: np.ndarray, preds: np.ndarray, class_names: Iterable[str]) -> None:
    cm = confusion_matrix(labels, preds)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix - Brain Tumor MRI")
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()


def _write_text_report(path: Path, labels: np.ndarray, preds: np.ndarray, class_names: Iterable[str]) -> None:
    report = classification_report(labels, preds, target_names=class_names)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(report)


def load_metadata(output_dir: Path) -> Dict[str, Any]:
    metadata_path = Path(output_dir) / METADATA_FILE
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata file not found at {metadata_path}")
    with open(metadata_path, "r", encoding="utf-8") as fp:
        return json.load(fp)


def summarize_dataset(data_dir: Path, img_size: int = DEFAULT_IMG_SIZE) -> Dict[str, Any]:
    """Compute simple statistics (class counts) for a dataset."""

    dataset = datasets.ImageFolder(data_dir, transform=transforms.Compose([transforms.Resize((img_size, img_size)), transforms.ToTensor()]))
    class_names = dataset.classes
    counts = {name: 0 for name in class_names}
    for _, label in dataset.samples:
        counts[class_names[label]] += 1
    summary = {
        "data_dir": str(data_dir),
        "class_names": class_names,
        "counts": counts,
        "total_images": len(dataset.samples),
    }
    return summary


def save_dataset_summary(output_dir: Path, summary: Dict[str, Any]) -> Path:
    output_dir = Path(output_dir)
    _ensure_dir(output_dir)
    summary_path = output_dir / DATASET_SUMMARY_FILE
    with open(summary_path, "w", encoding="utf-8") as fp:
        json.dump(summary, fp, indent=2)
    return summary_path


def clean_corrupted_images(data_dir: Path, write_report: bool = True) -> Dict[str, Any]:
    """Identify (and optionally remove) corrupted images to keep dataset healthy."""

    data_dir = Path(data_dir)
    removed: List[str] = []
    inspected = 0
    patterns = ("*.jpg", "*.jpeg", "*.png", "*.bmp")
    for pattern in patterns:
        for image_path in data_dir.rglob(pattern):
            inspected += 1
            try:
                with Image.open(image_path) as img:
                    img.verify()
            except (UnidentifiedImageError, OSError):
                removed.append(str(image_path))
                image_path.unlink(missing_ok=True)

    stats = {"inspected": inspected, "removed": len(removed), "removed_files": removed}
    if write_report:
        report_path = data_dir / "cleaning_report.json"
        with open(report_path, "w", encoding="utf-8") as fp:
            json.dump(stats, fp, indent=2)
        stats["report_path"] = str(report_path)
    return stats


def evaluate_best_model(
    data_dir: Path,
    output_dir: Path,
    batch_size: int = DEFAULT_BATCH_SIZE,
    num_workers: int = DEFAULT_NUM_WORKERS,
    img_size: int = DEFAULT_IMG_SIZE,
) -> Dict[str, Any]:
    """Reload best checkpoint and compute metrics on the test split."""

    metadata = load_metadata(output_dir)
    device = get_device()

    loaders = prepare_dataloaders(data_dir, batch_size, num_workers, img_size)
    model = create_model(len(loaders.class_names))
    model.load_state_dict(torch.load(metadata["best_model_path"], map_location=device))
    model = model.to(device)

    loss, acc, probs, preds, labels = evaluate_model(model, loaders.test, device)
    f1 = f1_score(labels, preds, average="weighted")
    iou = jaccard_score(labels, preds, average="weighted", labels=np.unique(preds))
    try:
        roc_auc = roc_auc_score(labels, probs, multi_class="ovr", average="weighted")
    except Exception:
        roc_auc = float("nan")

    metrics = {
        "loss": float(loss),
        "accuracy": float(acc),
        "f1_weighted": float(f1),
        "iou_weighted": float(iou),
        "roc_auc_weighted": float(roc_auc),
    }

    with open(Path(output_dir) / EVALUATION_FILE, "w", encoding="utf-8") as fp:
        json.dump(metrics, fp, indent=2)

    return metrics


def run_inference_samples(
    data_dir: Path,
    output_dir: Path,
    sample_count: int = 5,
    img_size: int = DEFAULT_IMG_SIZE,
) -> List[Dict[str, Any]]:
    """Run inference on a handful of samples for smoke testing."""

    metadata = load_metadata(output_dir)
    data_dir = Path(data_dir)
    device = get_device()

    _, val_tf = build_transforms(img_size)
    dataset = datasets.ImageFolder(data_dir, transform=val_tf)
    class_names = dataset.classes
    indices = list(range(len(dataset)))
    random.shuffle(indices)
    samples: List[Dict[str, Any]] = []

    model = create_model(len(class_names))
    model.load_state_dict(torch.load(metadata["best_model_path"], map_location=device))
    model = model.to(device)
    model.eval()

    for idx in indices[:sample_count]:
        image, label = dataset[idx]
        with torch.no_grad():
            logits = model(image.unsqueeze(0).to(device))
            probs = torch.softmax(logits, dim=1).cpu().numpy().flatten()
        prediction = int(np.argmax(probs))
        samples.append(
            {
                "index": idx,
                "actual": class_names[label],
                "predicted": class_names[prediction],
                "confidence": float(probs[prediction]),
                "probabilities": {class_names[i]: float(prob) for i, prob in enumerate(probs)},
            }
        )

    with open(Path(output_dir) / TEST_RESULTS_FILE, "w", encoding="utf-8") as fp:
        json.dump(samples, fp, indent=2)

    return samples


def ensure_data_dir(path: Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def ensure_output_dir(path: Path) -> Path:
    return ensure_data_dir(path)


def download_s3_prefix(bucket: str, prefix: str, destination: Path, client) -> Dict[str, Any]:
    """Download an entire S3 prefix into a destination folder."""

    destination = ensure_data_dir(destination)
    normalized_prefix = prefix.strip("/") if prefix else ""
    paginator = client.get_paginator("list_objects_v2")
    total_files = 0
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if key.endswith("/"):
                continue
            if normalized_prefix and key.startswith(normalized_prefix):
                rel_path = key[len(normalized_prefix) :].lstrip("/")
            else:
                rel_path = key
            target_path = destination / rel_path
            ensure_data_dir(target_path.parent)
            client.download_file(bucket, key, str(target_path))
            total_files += 1
    return {"bucket": bucket, "prefix": prefix, "destination": str(destination), "files_downloaded": total_files}


def resolve_local_data_root() -> Path:
    """Return the configured local data root directory."""

    root = Path(os.environ.get("LOCAL_DATA_DIR", "/opt/airflow/data/brain_tumor"))
    return ensure_data_dir(root)


def resolve_dataset_dir() -> Path:
    """Return the dataset directory that contains class sub-folders."""

    subdir = os.environ.get("LOCAL_DATA_SUBDIR")
    if subdir:
        return ensure_data_dir(resolve_local_data_root() / subdir)
    return resolve_local_data_root()


def resolve_output_dir() -> Path:
    """Return the training output directory for persisted artifacts."""

    output = Path(os.environ.get("TRAINING_OUTPUT_DIR", "/opt/airflow/output/brain_tumor"))
    return ensure_output_dir(output)
