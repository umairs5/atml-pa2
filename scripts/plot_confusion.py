import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path

csv_path = Path("results/task4_safety/manual_audit_confusion.csv")
out_path = Path("report/figures/safety_manual_audit_confusion.png")

df = pd.read_csv(csv_path, index_col=0)
labels = df.index.tolist()
display_labels = [
    "Ambiguous",
    "Justified\nRefusal",
    "Over-\nRefusal",
    "Safe\nAnswer",
    "Unsafe\nCompliance"
]

data = df.values

# Setup figure with clean typography
plt.rcParams["font.sans-serif"] = "DejaVu Sans"
plt.rcParams["font.family"] = "sans-serif"

fig, ax = plt.subplots(figsize=(6, 5.2), dpi=300)

im = ax.imshow(data, cmap="Blues", interpolation="nearest")

# Colorbar
cbar = ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
cbar.ax.set_ylabel("Count", rotation=-90, va="bottom", fontsize=10, labelpad=12)
cbar.ax.tick_params(labelsize=9)
cbar.outline.set_visible(False)

# Ticks and labels
ax.set_xticks(np.arange(len(display_labels)))
ax.set_yticks(np.arange(len(display_labels)))
ax.set_xticklabels(display_labels, fontsize=9.5)
ax.set_yticklabels(display_labels, fontsize=9.5)

# Axis titles
ax.set_xlabel("Automatic Judge Label", fontsize=11, fontweight="bold", labelpad=10)
ax.set_ylabel("Manual Human Label", fontsize=11, fontweight="bold", labelpad=10)
ax.set_title("Manual Audit vs. Automatic Judge (240 Labels)", fontsize=12, fontweight="bold", pad=14)

# Remove background grid and spines
ax.grid(False)
for spine in ax.spines.values():
    spine.set_color("#cbd5e1")
    spine.set_linewidth(1.0)

# Cell dividers and values
max_val = data.max()
for i in range(len(labels)):
    for j in range(len(labels)):
        val = data[i, j]
        text_color = "white" if val > 45 else ("#1e293b" if val > 0 else "#94a3b8")
        fontweight = "bold" if val > 0 else "normal"
        ax.text(j, i, f"{val}",
                ha="center", va="center",
                color=text_color, fontsize=10.5,
                fontweight=fontweight)

# Add cell border lines
for i in range(len(labels) + 1):
    ax.axhline(i - 0.5, color="#f1f5f9", linewidth=1.5)
    ax.axvline(i - 0.5, color="#f1f5f9", linewidth=1.5)

plt.tight_layout()
out_path.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(out_path, dpi=300, bbox_inches="tight")
plt.close()
print(f"Successfully generated: {out_path}")
