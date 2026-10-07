from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Polygon, Rectangle


OUTPUT_DIR = Path(__file__).resolve().parent


def feature_stack(ax, x, y, width, height, depth, color, label, detail):
    """Draw a compact pseudo-3D feature-map stack."""
    offset = min(0.018 * depth, 0.16)
    layers = min(max(depth // 32, 2), 6)
    for index in range(layers - 1, -1, -1):
        shift = index * offset / layers
        ax.add_patch(
            Rectangle(
                (x + shift, y + shift),
                width,
                height,
                facecolor=color,
                edgecolor="#17324d",
                linewidth=0.75,
                alpha=0.25 + 0.11 * (layers - index),
            )
        )
    ax.text(x + width / 2, y - 0.09, label, ha="center", va="top", fontsize=8.5, weight="bold")
    ax.text(x + width / 2, y - 0.19, detail, ha="center", va="top", fontsize=7.5, color="#334155")
    return x + width + offset


def vector_block(ax, x, y, width, height, color, label, detail):
    ax.add_patch(
        Rectangle((x, y), width, height, facecolor=color, edgecolor="#17324d", linewidth=0.9)
    )
    ax.text(x + width / 2, y + height / 2, label, ha="center", va="center", fontsize=8, weight="bold", rotation=90)
    ax.text(x + width / 2, y - 0.09, detail, ha="center", va="top", fontsize=7.5, color="#334155")
    return x + width


def arrow(ax, start, end, label=None, color="#334155"):
    ax.add_patch(
        FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=11, linewidth=1.1, color=color)
    )
    if label:
        ax.text((start[0] + end[0]) / 2, start[1] + 0.07, label, ha="center", va="bottom", fontsize=7, color=color)


fig, ax = plt.subplots(figsize=(15.5, 5.2))
fig.patch.set_facecolor("white")
ax.set_facecolor("white")

y = 0.42
centery = 0.70

# Input image: a neutral schematic face avoids implying that feature maps are real activations.
input_x, input_y, input_w, input_h = 0.15, 0.42, 0.64, 0.64
ax.add_patch(Rectangle((input_x, input_y), input_w, input_h, facecolor="#d8dee7", edgecolor="#17324d", linewidth=1.0))
ax.add_patch(plt.Circle((input_x + 0.32, input_y + 0.37), 0.20, facecolor="#eef2f7", edgecolor="#64748b", linewidth=1.0))
ax.plot([input_x + 0.25, input_x + 0.29], [input_y + 0.41, input_y + 0.41], color="#334155", linewidth=1.5)
ax.plot([input_x + 0.35, input_x + 0.39], [input_y + 0.41, input_y + 0.41], color="#334155", linewidth=1.5)
ax.plot([input_x + 0.27, input_x + 0.37], [input_y + 0.30, input_y + 0.30], color="#334155", linewidth=1.2)
ax.text(input_x + input_w / 2, input_y - 0.09, "Input face", ha="center", va="top", fontsize=8.5, weight="bold")
ax.text(input_x + input_w / 2, input_y - 0.19, "128×128×1", ha="center", va="top", fontsize=7.5, color="#334155")

positions = []
x = 1.15
stages = [
    (0.64, 0.64, 48, "Conv1 + MFM", "5×5 · 128×128×48", "#3b82f6"),
    (0.51, 0.51, 48, "Pool1", "2×2 · 64×64×48", "#22c55e"),
    (0.51, 0.51, 96, "Group1", "1×1 + 3×3 MFM · 64×64×96", "#f59e0b"),
    (0.39, 0.39, 96, "Pool2", "2×2 · 32×32×96", "#22c55e"),
    (0.39, 0.39, 192, "Group2", "1×1 + 3×3 MFM · 32×32×192", "#f59e0b"),
    (0.29, 0.29, 192, "Pool3", "2×2 · 16×16×192", "#22c55e"),
    (0.29, 0.29, 128, "Group3", "1×1 + 3×3 MFM · 16×16×128", "#f59e0b"),
    (0.29, 0.29, 128, "Group4", "1×1 + 3×3 MFM · 16×16×128", "#f59e0b"),
    (0.20, 0.20, 128, "Pool4", "2×2 · 8×8×128", "#22c55e"),
]

previous_end = input_x + input_w
for width, height, channels, label, detail, color in stages:
    current_y = centery - height / 2
    arrow(ax, (previous_end + 0.03, centery), (x - 0.06, centery))
    end = feature_stack(ax, x, current_y, width, height, channels, color, label, detail)
    positions.append((x, end, centery))
    previous_end = end
    x = end + 0.34

arrow(ax, (previous_end + 0.03, centery), (x - 0.06, centery), "flatten")
fc_end = vector_block(ax, x, 0.49, 0.18, 0.42, "#a855f7", "FC1 + MFM", "256-D embedding")

x2 = fc_end + 0.55
arrow(ax, (fc_end + 0.03, centery), (x2 - 0.06, centery), "dropout")
classifier_end = vector_block(ax, x2, 0.45, 0.18, 0.50, "#ef4444", "FC2", "identity logits")

x3 = classifier_end + 0.48
arrow(ax, (classifier_end + 0.03, centery), (x3 - 0.06, centery))
ax.add_patch(
    Polygon(
        [[x3, 0.48], [x3 + 0.36, 0.57], [x3 + 0.36, 0.83], [x3, 0.92]],
        closed=True,
        facecolor="#e2e8f0",
        edgecolor="#17324d",
        linewidth=0.9,
    )
)
ax.text(x3 + 0.18, 0.70, "Identity\nclasses", ha="center", va="center", fontsize=8, weight="bold")

ax.text(0.15, 1.40, "LightCNN-9 architecture", fontsize=14, weight="bold", color="#0f172a")
ax.text(
    0.15,
    1.26,
    "MFM doubles the intermediate channels and keeps the element-wise maximum of each channel pair.",
    fontsize=8.5,
    color="#475569",
)

# Legend
legend_y = 0.08
legend = [("#3b82f6", "initial convolution"), ("#f59e0b", "MFM convolution group"), ("#22c55e", "max pooling"), ("#a855f7", "embedding"), ("#ef4444", "classifier")]
legend_x = 0.15
for color, label in legend:
    ax.add_patch(Rectangle((legend_x, legend_y), 0.12, 0.06, facecolor=color, edgecolor="none"))
    ax.text(legend_x + 0.16, legend_y + 0.03, label, va="center", fontsize=7.5, color="#334155")
    legend_x += 1.24

ax.set_xlim(0, x3 + 0.55)
ax.set_ylim(0, 1.58)
ax.axis("off")
plt.tight_layout(pad=0.4)

for extension in ("png", "pdf", "svg"):
    fig.savefig(OUTPUT_DIR / f"lightcnn9_architecture.{extension}", dpi=300, bbox_inches="tight", facecolor="white")

plt.close(fig)
