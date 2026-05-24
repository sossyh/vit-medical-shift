import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from src.utils import NIH_LABELS


def compute_auc(logits, labels):
    """
    Compute per-class and mean AUC.
    Answers: how accurate is the model per disease?
    """
    probs  = torch.sigmoid(logits).numpy()
    labels = labels.numpy()

    results = {}
    for i, name in enumerate(NIH_LABELS):
        y = labels[:, i]
        if y.sum() == 0 or y.sum() == len(y):
            results[name] = float("nan")
            continue
        results[name] = roc_auc_score(y, probs[:, i])

    results["mean"] = float(np.nanmean(list(results.values())))
    return results


def compute_ece(logits, labels, n_bins=10):
    """
    Expected Calibration Error.
    Answers: when model says 80% confident, is it right 80% of the time?
    Lower ECE = better calibrated = more trustworthy.
    """
    probs  = torch.sigmoid(logits).numpy().flatten()
    labels = labels.numpy().flatten()
    bins   = np.linspace(0, 1, n_bins + 1)
    ece    = 0.0
    n      = len(probs)

    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (probs >= lo) & (probs < hi)
        if mask.sum() == 0:
            continue
        ece += (mask.sum() / n) * abs(probs[mask].mean() - labels[mask].mean())

    return float(ece)


def plot_reliability_diagram(logits, labels, title="Reliability Diagram",
                              save_path=None, n_bins=10):
    """
    Plot confidence vs accuracy.
    Perfect model = diagonal line.
    Above diagonal = underconfident.
    Below diagonal = overconfident.
    """
    probs     = torch.sigmoid(logits).numpy().flatten()
    labels_np = labels.numpy().flatten()
    bins      = np.linspace(0, 1, n_bins + 1)
    centers, accs = [], []

    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (probs >= lo) & (probs < hi)
        centers.append((lo + hi) / 2)
        accs.append(labels_np[mask].mean() if mask.sum() > 0 else 0)

    ece = compute_ece(logits, labels, n_bins)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.bar(centers, accs, width=0.9/n_bins, alpha=0.7,
           color="#378ADD", label="Model")
    ax.plot([0, 1], [0, 1], "k--", lw=1.5, label="Perfect calibration")
    ax.set_xlabel("Confidence")
    ax.set_ylabel("Accuracy")
    ax.set_title(f"{title}\nECE = {ece:.4f}")
    ax.legend()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150)
        print(f"Saved → {save_path}")
    plt.show()
    return fig


def temperature_scale(logits, labels, lr=0.01, n_iter=100):
    """
    Temperature scaling — simple post-hoc calibration fix.
    Finds best temperature T to divide logits by before sigmoid.
    Returns T value (T > 1 means model was overconfident).
    """
    T         = torch.nn.Parameter(torch.ones(1))
    optimizer = torch.optim.LBFGS([T], lr=lr, max_iter=n_iter)
    criterion = torch.nn.BCEWithLogitsLoss()

    def closure():
        optimizer.zero_grad()
        loss = criterion(logits / T, labels)
        loss.backward()
        return loss

    optimizer.step(closure)
    return T.item()


def print_metrics(auc_dict, ece, split_name=""):
    """Print a clean summary table of results."""
    label = f"[{split_name}] " if split_name else ""
    print(f"\n{label}Mean AUC : {auc_dict['mean']:.4f}")
    print(f"{label}ECE      : {ece:.4f}")
    print(f"\nPer-class AUC:")
    for k, v in auc_dict.items():
        if k != "mean":
            if np.isnan(v):
                print(f"  {k:<22} N/A")
            else:
                print(f"  {k:<22} {v:.4f}")