import numpy as np
import matplotlib.pyplot as plt


def scatter_density_plot(PR_true, PR_pred, vmax=None, save_path=None):
    PR_true = PR_true.values if hasattr(PR_true, "values") else PR_true
    PR_pred = PR_pred.values if hasattr(PR_pred, "values") else PR_pred

    mask = ~np.isnan(PR_true) & ~np.isnan(PR_pred)
    x = PR_true[mask].flatten()
    y = PR_pred[mask].flatten()

    plt.figure(figsize=(6, 6))
    hb = plt.hexbin(x, y, gridsize=100, cmap="viridis", bins="log", vmax=vmax)
    plt.plot([x.min(), x.max()], [x.min(), x.max()], "r--", linewidth=1)
    plt.xlabel("Observed (CPC)")
    plt.ylabel("Predicted (ML)")
    plt.title("Prediction vs Observation – Validation Set")
    cb = plt.colorbar(hb, fraction=0.046, pad=0.04)
    cb.set_label("log10(count)")

    lims = [min(x.min(), y.min()), max(x.max(), y.max())]
    plt.xlim(lims)
    plt.ylim(lims)
    plt.gca().set_aspect("equal", adjustable="box")

    plt.grid(True)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300)
    else:
        plt.show(block=True)
