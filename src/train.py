import os
import torch
import torch.nn as nn
from tqdm import tqdm
from src.utils import ensure_dir


def train_one_epoch(model, loader, optimizer, criterion, device, scheduler=None):
    """
    One full pass through the training data.
    Returns average loss for the epoch.
    """
    model.train()
    total_loss = 0.0

    for images, labels, _ in tqdm(loader, desc="Train", leave=False):
        images  = images.to(device)
        labels  = labels.to(device)

        optimizer.zero_grad()
        logits = model(images)
        loss   = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        if scheduler:
            scheduler.step()

        total_loss += loss.item() * images.size(0)

    return total_loss / len(loader.dataset)


@torch.no_grad()
def validate(model, loader, criterion, device):
    """
    Run model on validation set.
    Returns loss, all logits, all labels.
    """
    model.eval()
    total_loss = 0.0
    all_logits, all_labels = [], []

    for images, labels, _ in tqdm(loader, desc="Val  ", leave=False):
        images = images.to(device)
        labels = labels.to(device)

        logits = model(images)
        loss   = criterion(logits, labels)
        total_loss += loss.item() * images.size(0)

        all_logits.append(logits.cpu())
        all_labels.append(labels.cpu())

    return (
        total_loss / len(loader.dataset),
        torch.cat(all_logits),
        torch.cat(all_labels)
    )


def train(model, train_loader, val_loader, config, device):
    """
    Full training loop with checkpointing.
    Saves best model to results/checkpoints/.
    Returns history list of train/val losses per epoch.
    """
    cfg = config["training"]
    ensure_dir(cfg["checkpoint_dir"])

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["lr"],
        weight_decay=cfg["weight_decay"]
    )

    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=cfg["lr"],
        total_steps=cfg["epochs"] * len(train_loader),
        pct_start=cfg.get("warmup_epochs", 0) / cfg["epochs"],
    )

    criterion    = nn.BCEWithLogitsLoss()
    best_val_loss = float("inf")
    history      = []

    for epoch in range(1, cfg["epochs"] + 1):
        train_loss = train_one_epoch(
            model, train_loader, optimizer, criterion, device, scheduler
        )
        val_loss, val_logits, val_labels = validate(
            model, val_loader, criterion, device
        )

        history.append({
            "epoch"     : epoch,
            "train_loss": train_loss,
            "val_loss"  : val_loss
        })

        print(f"Epoch {epoch:02d} | train={train_loss:.4f} | val={val_loss:.4f}")

        # Save best checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            path = os.path.join(cfg["checkpoint_dir"], "best.pth")
            torch.save(model.state_dict(), path)
            print(f"  ✓ Saved best model → {path}")

    return history