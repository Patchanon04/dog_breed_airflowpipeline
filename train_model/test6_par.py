#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Brain Tumor MRI Classification (PyTorch)
----------------------------------------
- Dataset: ImageFolder (subfolders = class names)
- Model: ResNet18 pretrained on ImageNet
- Features: EarlyStopping, TensorBoard logging
- Saves best model and evaluation results

Folders:
  📁 Model → DE_MSc/notebook/model
  📁 Logs  → DE_MSc/notebook/log

Run:
python brain_tumor_classification.py --data_dir "N:/test_MSc/DE_MSc/data/all_dataset_v4" --epochs 60
"""

import os
import time
import argparse
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms, models
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, f1_score, roc_auc_score, jaccard_score
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# ========== SETUP ==========
def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def set_seed(seed=42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ========== DATA ==========
def build_transforms(img_size=224):
    mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    train_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    val_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    return train_tf, val_tf


def split_dataset(dataset, test_size=0.1, val_ratio=1/9):
    targets = dataset.targets
    indices = list(range(len(targets)))
    train_val_idx, test_idx = train_test_split(indices, test_size=test_size, stratify=targets, random_state=42)
    train_val_targets = [targets[i] for i in train_val_idx]
    train_idx, val_idx = train_test_split(train_val_idx, test_size=val_ratio, stratify=train_val_targets, random_state=42)
    return train_idx, val_idx, test_idx


# ========== MODEL ==========
def create_model(num_classes):
    model = models.resnet18(pretrained=True)
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(in_features, num_classes)
    )
    return model


# ========== EVALUATION ==========
@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    criterion = nn.CrossEntropyLoss()
    total_loss, total_correct = 0.0, 0
    all_preds, all_labels, all_probs = [], [], []

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
    return avg_loss, avg_acc, np.concatenate(all_probs), np.concatenate(all_preds), np.concatenate(all_labels)


# ========== TRAINING LOOP ==========
def train(args):
    set_seed()
    device = get_device()
    print(f"Using device: {device}")

    data_dir = Path(args.data_dir)
    model_dir = Path("DE_MSc/notebook/model")
    log_dir   = Path("DE_MSc/notebook/log")

    model_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    train_tf, val_tf = build_transforms()
    dataset_train_like = datasets.ImageFolder(data_dir, transform=train_tf)
    dataset_val_like = datasets.ImageFolder(data_dir, transform=val_tf)
    class_names = dataset_train_like.classes
    print("Classes:", class_names)

    train_idx, val_idx, test_idx = split_dataset(dataset_train_like)
    train_ds = Subset(dataset_train_like, train_idx)
    val_ds   = Subset(dataset_val_like, val_idx)
    test_ds  = Subset(dataset_val_like, test_idx)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader   = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=4)
    test_loader  = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=4)

    model = create_model(len(class_names)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)

    best_val_acc = 0.0
    no_improve = 0

    start_time = time.time()
    for epoch in range(args.epochs):
        model.train()
        train_loss, train_correct = 0.0, 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * inputs.size(0)
            preds = outputs.argmax(1)
            train_correct += torch.sum(preds == labels).item()

        train_loss /= len(train_ds)
        train_acc = train_correct / len(train_ds)
        val_loss, val_acc, _, _, _ = evaluate(model, val_loader, device)

        print(f"Epoch {epoch+1}/{args.epochs} | "
              f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            no_improve = 0
            torch.save(model.state_dict(), model_dir / "best_model.pth")
        else:
            no_improve += 1
            if no_improve >= args.patience:
                print("⏹ Early stopping triggered.")
                break

    total_min = (time.time() - start_time) / 60
    print(f"Training complete in {total_min:.2f} minutes")

    # ========== TEST EVALUATION ==========
    model.load_state_dict(torch.load(model_dir / "best_model.pth"))
    test_loss, test_acc, probs, preds, labels = evaluate(model, test_loader, device)
    print(f"✅ Test Accuracy: {test_acc:.4f}")

    f1 = f1_score(labels, preds, average="weighted")
    iou = jaccard_score(labels, preds, average="weighted", labels=np.unique(preds))
    try:
        roc_auc = roc_auc_score(labels, probs, multi_class="ovr", average="weighted")
    except Exception:
        roc_auc = float("nan")

    print(f"F1: {f1:.4f} | IoU: {iou:.4f} | ROC-AUC: {roc_auc:.4f}")

    # Confusion Matrix
    cm = confusion_matrix(labels, preds)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix - Brain Tumor MRI")
    plt.tight_layout()
    plt.savefig(log_dir / "confusion_matrix.png", dpi=300)
    plt.close()

    # Save classification report
    report_path = log_dir / "report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(classification_report(labels, preds, target_names=class_names))

    print(f"\n📊 Confusion matrix and report saved to: {log_dir}")
    print(classification_report(labels, preds, target_names=class_names))


# ========== MAIN ==========
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=5)
    args = parser.parse_args()
    train(args)
