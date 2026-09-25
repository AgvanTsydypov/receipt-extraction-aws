"""Regenerate the README charts. Numbers are copied from results/ summaries and confidence reports."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})

# Chart 1: cost vs header accuracy on the dev split
points = [
    ("Nova 2 Lite, image", 0.791, 0.79),
    ("Claude Haiku 4.5, image", 3.346, 0.86),
    ("Textract AnalyzeExpense", 10.0, 0.87),
    ("Nova 2 Lite, hybrid", 10.811, 0.82),
    ("Claude Haiku 4.5, hybrid", 13.426, 0.91),
]
fig, ax = plt.subplots(figsize=(8, 5))
for name, cost, acc in points:
    ax.scatter(cost, acc, s=90, zorder=3)
    dx, ha = (1.12, "left") if cost < 10 else (0.9, "right")
    ax.annotate(name, (cost, acc), xytext=(cost * dx, acc + 0.006), ha=ha, fontsize=10)
ax.set_xscale("log")
ax.set_xticks([0.5, 1, 2, 5, 10, 20])
ax.set_xticklabels(["$0.5", "$1", "$2", "$5", "$10", "$20"])
ax.minorticks_off()
ax.set_xlabel("Cost per 1,000 receipts, USD (log scale)")
ax.set_ylabel("Header accuracy (all amounts correct)")
ax.set_title("Cost vs accuracy, dev split (100 receipts)")
ax.set_ylim(0.75, 0.94)
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig("docs/cost_vs_accuracy.png", dpi=150)

# Chart 2: straight-through processing vs precision (5-fold CV on 800 train receipts)
fig, ax = plt.subplots(figsize=(8, 5))
gb = [(0.951, 0.789), (0.971, 0.606), (0.982, 0.475), (1.000, 0.059)]
ax.plot([p for p, _ in gb], [s for _, s in gb], marker="o", lw=2, color="#1f77b4",
        label="Gradient boosting confidence model")
for p, s in gb:
    offset, ha = ((-10, 6), "right") if p >= 0.999 else ((8, 4), "left")
    ax.annotate(f"{s:.0%} auto at {p:.1%}", (p, s), xytext=offset, textcoords="offset points",
                ha=ha, fontsize=9)
ax.scatter([0.938], [0.843], marker="s", s=80, zorder=3, color="#d62728",
           label="Rule: Claude and Textract agree")
ax.scatter([0.858], [1.0], marker="^", s=90, zorder=3, color="#7f7f7f", label="Accept everything")
ax.set_xlabel("Precision of auto-approved receipts")
ax.set_ylabel("Share processed without human review")
ax.set_title("Confidence-based routing (5-fold CV, 800 receipts)")
ax.set_xlim(0.85, 1.005)
ax.set_ylim(0, 1.05)
ax.grid(alpha=0.3)
ax.legend(loc="lower left", frameon=False)
fig.tight_layout()
fig.savefig("docs/stp_vs_precision.png", dpi=150)
print("charts saved")
