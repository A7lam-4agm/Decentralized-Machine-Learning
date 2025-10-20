"""
CS 595 – Assignment 2: Federated Learning Simulation (FedAvg)

What this script does:
- Uses ResNet-18 adapted for CIFAR-10 (32×32)
- Creates non-IID client partitions via Dirichlet(alpha)
- Trains local clients (Colab-safe: sequential on one GPU)
- Logs per-batch rows (time, round, batch_num, client_id, train_loss, train_acc)
- Logs per-round metrics (test_loss, test_acc, selected_clients)
- Saves client×class distribution matrix (CSV) for the non-IID heatmap
- Auto-stops when test accuracy ≥ 70% (per professor requirement)
- Saves final global model as global_model.pth
- Generates learning-curve & distribution plots

Author: Ahlam Abu Mismar
"""

from __future__ import annotations

import os
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, models, transforms


# =========================
# Configuration (no magic numbers)
# =========================

@dataclass(frozen=True)
class Config:
    seed: int = 42
    data_root: str = "./data"
    results_dir: str = "./results"
    num_classes: int = 10               # CIFAR-10
    num_clients: int = 64
    clients_per_round: int = 12         # Colab-safe (reduce if OOM)
    rounds: int = 80                    # we will stop early at 70%
    local_epochs: int = 3
    batch_size: int = 64                # use 32 if OOM
    learning_rate: float = 0.05         # stabler for FL than 0.1
    momentum: float = 0.9
    weight_decay: float = 5e-4          # strong CIFAR-10 default
    dirichlet_alpha: float = 10.0       # near-IID → easier convergence
    max_workers: int = 1                # IMPORTANT: sequential on one GPU


CFG = Config()
os.makedirs(CFG.results_dir, exist_ok=True)

# Required outputs per the assignment
BATCH_LOG_PATH = os.path.join(CFG.results_dir, "training_results.csv")   # per-batch
ROUND_LOG_PATH = os.path.join(CFG.results_dir, "round_metrics.csv")      # per-round
DIST_CSV_PATH  = os.path.join(CFG.results_dir, "client_distribution.csv")# clients×classes
FINAL_MODEL_PATH = "global_model.pth"


# =========================
# Utilities
# =========================

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

def now_iso() -> str:
    return datetime.utcnow().isoformat()


# =========================
# Data loading & partitioning
# =========================

def build_transforms() -> tuple[transforms.Compose, transforms.Compose]:
    train_tf = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2470, 0.2435, 0.2616)),
    ])
    test_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2470, 0.2435, 0.2616)),
    ])
    return train_tf, test_tf

def load_cifar10(root: str) -> tuple[datasets.CIFAR10, datasets.CIFAR10]:
    train_tf, test_tf = build_transforms()
    train_ds = datasets.CIFAR10(root, train=True,  download=True, transform=train_tf)
    test_ds  = datasets.CIFAR10(root, train=False, download=True, transform=test_tf)
    return train_ds, test_ds

def dirichlet_partition(
    labels: np.ndarray,
    num_clients: int,
    num_classes: int,
    alpha: float,
) -> dict[int, np.ndarray]:
    """
    Non-IID label skew using Dirichlet(alpha) per class.
    Returns: {client_id: array(indices)}
    """
    rng = np.random.default_rng(CFG.seed)
    idx_by_class = [np.where(labels == c)[0] for c in range(num_classes)]
    for c in range(num_classes):
        rng.shuffle(idx_by_class[c])

    client_indices: dict[int, list[int]] = {cid: [] for cid in range(num_clients)}

    for c in range(num_classes):
        proportions = rng.dirichlet(alpha * np.ones(num_clients))
        cls_idx = idx_by_class[c]
        split_points = (np.cumsum(proportions / proportions.sum() * len(cls_idx)).astype(int))[:-1]
        chunks = np.split(cls_idx, split_points)
        for cid, chunk in enumerate(chunks):
            client_indices[cid].extend(chunk.tolist())

    for cid in client_indices:
        rng.shuffle(client_indices[cid])
        client_indices[cid] = np.asarray(client_indices[cid], dtype=int)

    return client_indices

def save_distribution_csv(
    client_indices: dict[int, np.ndarray],
    labels: np.ndarray,
    num_classes: int,
    out_csv: str,
) -> None:
    mat = np.zeros((len(client_indices), num_classes), dtype=int)
    for cid, idx in client_indices.items():
        hist = np.bincount(labels[idx], minlength=num_classes)
        mat[cid] = hist
    df = pd.DataFrame(mat, columns=[f"class_{c}" for c in range(num_classes)])
    df.index.name = "client_id"
    df.to_csv(out_csv, index=True)


# =========================
# Model helpers (ResNet-18 adapted to CIFAR-10)
# =========================

def build_model(num_classes: int) -> nn.Module:
    """ImageNet backbone adapted to 32×32 + CIFAR-10 head."""
    # Torchvision API compatibility (weights vs pretrained)
    try:
        m = models.resnet18(weights=None)
    except TypeError:
        m = models.resnet18(pretrained=False)
    # CIFAR-10: use smaller first conv + no initial maxpool
    m.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    m.maxpool = nn.Identity()
    m.fc = nn.Linear(m.fc.in_features, num_classes)
    return m

def get_weights(model: nn.Module) -> Dict[str, torch.Tensor]:
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

def set_weights(model: nn.Module, weights: Dict[str, torch.Tensor]) -> None:
    sd = model.state_dict()
    for k in sd.keys():
        sd[k].copy_(weights[k].to(sd[k].device))

def average_weights_weighted(weight_list: List[Dict[str, torch.Tensor]], client_sizes: List[int]) -> Dict[str, torch.Tensor]:
    """FedAvg weighted by number of samples per client."""
    total_size = sum(client_sizes)
    out: Dict[str, torch.Tensor] = {}
    for key in weight_list[0].keys():
        acc = torch.zeros_like(weight_list[0][key])
        for i, w in enumerate(weight_list):
            acc += w[key] * client_sizes[i]
        out[key] = acc / total_size
    return out


# =========================
# Logging helpers
# =========================

def log_batch_row(round_id: int, batch_idx: int, client_id: int, train_loss: float, train_acc_percent: float) -> None:
    row = {
        "time": now_iso(),
        "round": round_id,
        "batch_num": batch_idx,
        "client_id": client_id,
        "train_loss": float(train_loss),
        "train_acc": float(train_acc_percent),
    }
    pd.DataFrame([row]).to_csv(
        BATCH_LOG_PATH, mode="a", header=not os.path.exists(BATCH_LOG_PATH), index=False
    )

def log_round_row(round_id: int, test_loss: float, test_acc_percent: float, selected_clients: list[int]) -> None:
    row = {
        "time": now_iso(),
        "round": round_id,
        "test_loss": float(test_loss),
        "test_acc": float(test_acc_percent),
        "clients_selected": ",".join(map(str, selected_clients)),
    }
    pd.DataFrame([row]).to_csv(
        ROUND_LOG_PATH, mode="a", header=not os.path.exists(ROUND_LOG_PATH), index=False
    )


# =========================
# Local train job (runs sequentially on Colab)
# =========================

def local_train_job(
    client_id: int,
    init_weights: Dict[str, torch.Tensor],
    subset: Subset,
    device: torch.device,
    round_id: int,
) -> Dict[str, torch.Tensor]:
    """
    Train one client's local model for CFG.local_epochs on their private subset.
    Returns updated weights to aggregate.
    """
    model = build_model(CFG.num_classes).to(device)
    set_weights(model, init_weights)
    model.train()

    # dataloader tuned for Colab
    loader = DataLoader(
        subset,
        batch_size=CFG.batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=(device.type == "cuda"),
    )

    # cosine LR schedule over global rounds (smooths optimization)
    t = (round_id - 1) / max(CFG.rounds - 1, 1)
    current_lr = 0.5 * CFG.learning_rate * (1 + np.cos(np.pi * t))

    opt = torch.optim.SGD(
        model.parameters(),
        lr=current_lr,
        momentum=CFG.momentum,
        weight_decay=CFG.weight_decay,
    )

    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True

    for epoch in range(CFG.local_epochs):
        for batch_idx, (images, targets) in enumerate(loader):
            images, targets = images.to(device), targets.to(device)
            opt.zero_grad()
            logits = model(images)
            # small label smoothing helps 1–2% on CIFAR-10; remove if not allowed
            loss = F.cross_entropy(logits, targets, label_smoothing=0.05)
            loss.backward()
            opt.step()

            with torch.no_grad():
                acc = (logits.argmax(1) == targets).float().mean().item() * 100.0

            log_batch_row(round_id, batch_idx, client_id, loss.item(), acc)

    return get_weights(model)


# =========================
# Evaluation
# =========================

@torch.no_grad()
def evaluate(model: nn.Module, test_ds, device: torch.device) -> Tuple[float, float]:
    model.eval()
    loader = DataLoader(
        test_ds, batch_size=256, shuffle=False, num_workers=2, pin_memory=(device.type == "cuda")
    )
    n, correct, loss_sum = 0, 0, 0.0
    for images, targets in loader:
        images, targets = images.to(device), targets.to(device)
        logits = model(images)
        loss_sum += F.cross_entropy(logits, targets, reduction="sum").item()
        correct += (logits.argmax(1) == targets).sum().item()
        n += targets.size(0)
    avg_loss = loss_sum / max(n, 1)
    acc = (correct / max(n, 1)) * 100.0
    return avg_loss, acc


# =========================
# Visualization for Report
# =========================

def generate_report_plots():
    """Generate publication-quality plots for report."""
    import matplotlib.pyplot as plt
    import seaborn as sns

    plt.style.use("seaborn-v0_8")
    sns.set_palette("husl")

    # Plot 1: Learning Curve
    round_df = pd.read_csv(ROUND_LOG_PATH)
    plt.figure(figsize=(10, 6))
    plt.plot(round_df['round'], round_df['test_acc'],
             linewidth=3, label='Test Accuracy', marker='o', markersize=4)
    plt.axhline(y=70, color='red', linestyle='--', linewidth=2,
                label='Target 70% Accuracy', alpha=0.8)
    achievement_round = round_df[round_df['test_acc'] >= 70]['round'].min()
    if not pd.isna(achievement_round):
        plt.axvline(x=achievement_round, color='green', linestyle=':',
                    alpha=0.7, label=f'70% at Round {int(achievement_round)}')
    plt.xlabel('Communication Rounds', fontsize=12)
    plt.ylabel('Test Accuracy (%)', fontsize=12)
    plt.title('Federated Learning Performance - CIFAR-10 (Optimized)', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 100)
    plt.savefig('learning_curve_optimized.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()

    # Plot 2: Data Distribution Heatmap
    dist_df = pd.read_csv(DIST_CSV_PATH, index_col='client_id')
    plt.figure(figsize=(14, 8))
    sns.heatmap(dist_df, cmap='YlOrRd', cbar_kws={'label': 'Sample Count'},
                annot=False, linewidths=0.5)
    plt.title(f'Non-IID Data Distribution (Dirichlet α={CFG.dirichlet_alpha}) across {CFG.num_clients} Clients',
              fontsize=14, fontweight='bold')
    plt.xlabel('CIFAR-10 Class Label', fontsize=12)
    plt.ylabel('Client ID', fontsize=12)
    plt.savefig('data_distribution_optimized.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()

    # Summary print
    final_acc = round_df['test_acc'].max()
    print("🎯 FINAL PERFORMANCE SUMMARY")
    print(f"   Peak Test Accuracy: {final_acc:.2f}%")
    print(f"   Rounds logged: {round_df['round'].max()}")


# =========================
# Main FedAvg loop
# =========================

def main() -> None:
    set_seed(CFG.seed)
    device = get_device()
    print(f"[Info] Using device: {device}")

    # 1) Load data and create non-IID client splits
    train_ds, test_ds = load_cifar10(CFG.data_root)
    labels = np.array(train_ds.targets, dtype=int)
    client_map = dirichlet_partition(labels, CFG.num_clients, CFG.num_classes, CFG.dirichlet_alpha)
    save_distribution_csv(client_map, labels, CFG.num_classes, DIST_CSV_PATH)
    print(f"[Info] Saved client distribution matrix -> {DIST_CSV_PATH}")

    client_subsets = {cid: Subset(train_ds, idxs) for cid, idxs in client_map.items()}

    # 2) Initialize global model
    global_model = build_model(CFG.num_classes).to(device)

    # Optional baseline row (round 0)
    base_loss, base_acc = evaluate(global_model, test_ds, device)
    log_round_row(0, base_loss, base_acc, [])
    print(f"[Baseline] Test acc: {base_acc:.2f}%")

    # 3) FedAvg communication rounds
    for r in range(1, CFG.rounds + 1):
        print(f"\n=== Communication Round {r}/{CFG.rounds} ===")

        # sample clients (random, size-balanced on average with alpha=10)
        selected = sorted(random.sample(range(CFG.num_clients), CFG.clients_per_round))
        print(f"[Info] Selected clients: {selected}")

        # broadcast weights
        init_w = get_weights(global_model)

        # local updates (sequential on Colab via max_workers=1)
        local_weights: List[Dict[str, torch.Tensor]] = []
        with ThreadPoolExecutor(max_workers=CFG.max_workers) as ex:
            futs = {
                ex.submit(local_train_job, cid, init_w, client_subsets[cid], device, r): cid
                for cid in selected
            }
            for fut in as_completed(futs):
                cid = futs[fut]
                try:
                    local_weights.append(fut.result())
                except Exception as e:
                    print(f"[Warn] Client {cid} failed: {e}")

        # aggregate (Weighted FedAvg)
        if local_weights:
            client_sizes = [len(client_subsets[cid]) for cid in selected]
            set_weights(global_model, average_weights_weighted(local_weights, client_sizes))

        # evaluate on test set and log per-round metrics
        test_loss, test_acc = evaluate(global_model, test_ds, device)
        print(f"[Round {r}] Test loss: {test_loss:.4f} | Test acc: {test_acc:.2f}%")
        log_round_row(r, test_loss, test_acc, selected)

        # ----- EARLY STOP AT 70% -----
        if test_acc >= 70.0:
            print(f"🎉 TARGET REACHED: {test_acc:.2f}% accuracy at round {r} — stopping.")
            torch.save(global_model.state_dict(), FINAL_MODEL_PATH)
            print(f"[Done] Saved final global model -> {FINAL_MODEL_PATH}")
            generate_report_plots()
            return

    # 4) Save final model and plots if we didn’t early-stop
    torch.save(global_model.state_dict(), FINAL_MODEL_PATH)
    print(f"\n[Done] Saved final global model -> {FINAL_MODEL_PATH}")
    print(f"[Done] Per-batch logs   -> {BATCH_LOG_PATH}")
    print(f"[Done] Per-round logs   -> {ROUND_LOG_PATH}")
    print(f"[Done] Distribution CSV -> {DIST_CSV_PATH}")
    generate_report_plots()


if __name__ == "__main__":
    main()
