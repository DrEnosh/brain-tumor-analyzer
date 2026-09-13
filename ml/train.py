import argparse
import json
import random
import time

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score
from torch import nn
from torch.utils.data import DataLoader

from ml.config import ARCHITECTURES, CLASSES, MODELS_DIR, SEED
from ml.dataset import MRIDataset
from ml.models import build_model, is_pretrained_backbone
from ml.preprocessing import IMAGENET_MEAN, IMAGENET_STD, IMG_SIZE

DEFAULTS = {
    "baseline_cnn": {"epochs": 40, "lr": 1e-3, "patience": 8},
    "efficientnet_b0": {"epochs": 20, "lr": 1e-3, "patience": 5},
    "resnet50": {"epochs": 20, "lr": 1e-3, "patience": 5},
}


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def make_optimizer(model, arch: str, lr: float, weight_decay: float):
    if not is_pretrained_backbone(arch):
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    head = list(model.fc.parameters())
    head_ids = {id(p) for p in head}
    backbone = [p for p in model.parameters() if id(p) not in head_ids]
    return torch.optim.AdamW(
        [{"params": backbone, "lr": lr * 0.1}, {"params": head, "lr": lr}], weight_decay=weight_decay)


@torch.no_grad()
def predict(model, loader, device, amp: bool = True):
    model.eval()
    logits, targets = [], []
    for x, y in loader:
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp and device.type == "cuda"):
            out = model(x.to(device, non_blocking=True))
        logits.append(out.float().cpu())
        targets.append(y)
    return torch.cat(logits), torch.cat(targets)


def run(arch: str, epochs: int, lr: float, patience: int, batch_size: int, weight_decay: float,
        workers: int, seed: int, limit_batches: int = 0) -> dict:
    seed_everything(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = MODELS_DIR / arch
    out_dir.mkdir(parents=True, exist_ok=True)

    generator = torch.Generator().manual_seed(seed)
    loader_kwargs = dict(batch_size=batch_size, num_workers=workers, pin_memory=device.type == "cuda",
                         persistent_workers=workers > 0)
    train_loader = DataLoader(MRIDataset("train", augment=True), shuffle=True, generator=generator,
                              drop_last=True, **loader_kwargs)
    val_loader = DataLoader(MRIDataset("val"), shuffle=False, **loader_kwargs)

    model = build_model(arch, pretrained=True).to(device)
    optimizer = make_optimizer(model, arch, lr, weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    scaler = torch.amp.GradScaler(device.type, enabled=device.type == "cuda")

    history, best_f1, best_epoch, stale = [], -1.0, 0, 0
    for epoch in range(1, epochs + 1):
        model.train()
        start, running, seen = time.time(), 0.0, 0
        for step, (x, y) in enumerate(train_loader):
            if limit_batches and step >= limit_batches:
                break
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                loss = criterion(model(x), y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running += loss.item() * len(y)
            seen += len(y)
        scheduler.step()
        train_seconds = time.time() - start

        val_logits, val_targets = predict(model, val_loader, device)
        val_loss = criterion(val_logits, val_targets).item()
        val_pred = val_logits.argmax(1).numpy()
        val_acc = accuracy_score(val_targets.numpy(), val_pred)
        val_f1 = f1_score(val_targets.numpy(), val_pred, average="macro")
        record = {"epoch": epoch, "train_loss": running / max(seen, 1), "val_loss": val_loss,
                  "val_accuracy": val_acc, "val_macro_f1": val_f1, "train_seconds": round(train_seconds, 1),
                  "val_seconds": round(time.time() - start - train_seconds, 1)}
        history.append(record)
        print(json.dumps({"arch": arch, **record}), flush=True)

        if val_f1 > best_f1:
            best_f1, best_epoch, stale = val_f1, epoch, 0
            torch.save({
                "arch": arch, "state_dict": model.state_dict(), "classes": list(CLASSES), "epoch": epoch,
                "val_macro_f1": val_f1, "val_accuracy": val_acc, "img_size": IMG_SIZE,
                "mean": IMAGENET_MEAN.tolist(), "std": IMAGENET_STD.tolist(), "seed": seed,
            }, out_dir / "best.pt")
        else:
            stale += 1
            if stale >= patience:
                print(f"early stopping at epoch {epoch} (best epoch {best_epoch})", flush=True)
                break

    summary = {"arch": arch, "best_epoch": best_epoch, "best_val_macro_f1": best_f1, "epochs_run": len(history),
               "hyperparameters": {"epochs": epochs, "lr": lr, "patience": patience, "batch_size": batch_size,
                                   "weight_decay": weight_decay, "seed": seed, "label_smoothing": 0.1}}
    (out_dir / "history.json").write_text(json.dumps({"summary": summary, "history": history}, indent=2))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a brain-tumour MRI classifier.")
    parser.add_argument("--arch", choices=ARCHITECTURES, required=True)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--patience", type=int)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--limit-batches", type=int, default=0, help="smoke-test: stop each epoch early")
    args = parser.parse_args()
    d = DEFAULTS[args.arch]
    print(run(args.arch, args.epochs or d["epochs"], args.lr or d["lr"], args.patience or d["patience"],
              args.batch_size, args.weight_decay, args.workers, args.seed, args.limit_batches))


if __name__ == "__main__":
    main()
