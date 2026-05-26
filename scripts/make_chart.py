"""
Regenerates the two README charts from the benchmark results.

Numbers here are the source of truth - if you change them, also update the
results table in README.md so the two stay in sync.

Outputs:
    assets/training_time_comparison.png   bar chart, time per implementation
    assets/tradeoff_scatter.png           scatter: dev effort (LOC) vs runtime
"""

import os

import matplotlib.pyplot as plt

# --- Benchmark data ---------------------------------------------------------
# Lines of code are counted as "non-blank, non-comment" lines via:
#   grep -cE '^[[:space:]]*[^[:space:]/#*]' src/<file>
# The C++/CUDA implementations share csv_loader.hpp (64 LOC), which is added
# to each of their counts below.
IMPLEMENTATIONS = [
    # name,             train_s,  acc,    code_loc, color
    ("CPU Baseline",     0.70,    0.9946,    58,    "#888888"),
    ("RAPIDS",           0.45,    0.9989,    53,    "#76b900"),
    ("OpenACC",         14.70,    0.9946,   144,    "#0d72b9"),  # 80 + 64
    ("CUDA Single-GPU",  0.88,    0.9946,   157,    "#76b900"),  # 93 + 64
    ("CUDA Multi-GPU",   0.50,    0.9946,   243,    "#2b7a3a"),  # 179 + 64
]


def ensure_assets_dir() -> None:
    os.makedirs("assets", exist_ok=True)


def chart_training_time() -> None:
    names  = [row[0] for row in IMPLEMENTATIONS]
    times  = [row[1] for row in IMPLEMENTATIONS]
    colors = [row[4] for row in IMPLEMENTATIONS]

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(names, times, color=colors, edgecolor="black", linewidth=0.6)

    ax.set_ylabel("Training time (seconds, lower is better)", fontsize=11)
    ax.set_title(
        "Training Time: Logistic Regression on Credit Card Fraud "
        "(284,807 samples, 100 epochs)",
        fontsize=12,
    )
    ax.set_axisbelow(True)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.set_ylim(0, max(times) * 1.15)

    for bar, val in zip(bars, times):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(times) * 0.015,
            f"{val:.2f}s",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )

    plt.tight_layout()
    plt.savefig("assets/training_time_comparison.png", dpi=140)
    plt.close(fig)
    print("Saved assets/training_time_comparison.png")


def chart_tradeoff_scatter() -> None:
    """Dev effort (lines of code) vs. runtime. Lower-left is the sweet spot.

    Y-axis is log-scaled so the four sub-second implementations are
    distinguishable instead of squashed onto the x-axis by the OpenACC point.
    """
    fig, ax = plt.subplots(figsize=(9.5, 5.5))

    for name, train_s, _acc, loc, color in IMPLEMENTATIONS:
        ax.scatter(loc, train_s, s=280, c=color, edgecolor="black",
                   linewidth=0.9, zorder=3)

    # Hand-tuned label positions to avoid marker overlap on a log y-axis.
    # Each entry: (xytext_x, xytext_y, ha, va)
    label_positions = {
        "CPU Baseline":    (75,  0.72, "left",   "center"),
        "RAPIDS":          (75,  0.42, "left",   "center"),
        "OpenACC":         (160, 14.7, "left",   "center"),
        "CUDA Single-GPU": (175, 0.88, "left",   "center"),
        "CUDA Multi-GPU":  (235, 0.38, "right",  "center"),
    }
    for name, train_s, _acc, loc, _color in IMPLEMENTATIONS:
        lx, ly, ha, va = label_positions[name]
        ax.annotate(name, xy=(loc, train_s), xytext=(lx, ly),
                    fontsize=10, fontweight="bold", ha=ha, va=va)

    ax.set_xlabel("Lines of code  →  more developer effort", fontsize=11)
    ax.set_ylabel("Training time, seconds  →  slower (log scale)", fontsize=11)
    ax.set_title(
        "Productivity vs. Performance — lower-left is the sweet spot",
        fontsize=12,
    )
    ax.set_yscale("log")
    ax.set_axisbelow(True)
    ax.grid(which="both", linestyle="--", alpha=0.4)

    # Shade the "sweet spot" region (low LOC, sub-second runtime).
    ax.axhspan(0.3, 1.0, xmin=0.02, xmax=0.40, alpha=0.10, color="green",
               zorder=0)

    ax.set_xlim(0, 280)
    ax.set_ylim(0.3, 25)

    plt.tight_layout()
    plt.savefig("assets/tradeoff_scatter.png", dpi=140)
    plt.close(fig)
    print("Saved assets/tradeoff_scatter.png")


def main() -> None:
    ensure_assets_dir()
    chart_training_time()
    chart_tradeoff_scatter()


if __name__ == "__main__":
    main()
