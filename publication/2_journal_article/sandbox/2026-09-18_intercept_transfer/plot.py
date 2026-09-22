"""Learning-curve figure: Beta energy precision and Energy IoU against reviewed days, from curve_summary.csv."""

from __future__ import annotations

import pathlib

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = pathlib.Path(__file__).resolve().parent
NAVY, ORANGE, GREY, RED, LIGHT = "#223041", "#F39A24", "#7C858F", "#B64A4A", "#5C7D99"


def main() -> None:
    s = pd.read_csv(HERE / "curve_summary.csv")
    curve = pd.read_csv(HERE / "curve.csv")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, direction, gate in zip(axes, ("alpha->beta", "beta->alpha"), (0.90, None), strict=True):
        d = s[s["direction"] == direction]
        ref = curve[(curve["direction"] == direction) & (curve["level"] == "R0_within_cohort")].iloc[0]
        k0 = curve[(curve["direction"] == direction) & (curve["level"] == "k0_source_pair")].iloc[0]
        for level, design, colour, ls in (("population", "random", NAVY, "-"), ("population", "review", ORANGE, "-"),
                                          ("station", "random", NAVY, "--"), ("station", "review", ORANGE, "--")):
            g = d[(d["level"] == level) & (d["design"] == design)].sort_values("k")
            ax.plot(g["k"], g["energy_precision_med"], marker="o", color=colour, ls=ls, label=f"{level}, {design} sample")
            ax.fill_between(g["k"], g["energy_precision_p05"], g["energy_precision_p95"], color=colour, alpha=0.08)
        ax.axhline(ref["energy_precision"], color=GREY, lw=1, ls=":", label="within-cohort reference")
        ax.axhline(k0["energy_precision"], color=RED, lw=1, ls=":", label="foreign pair, no review (k = 0)")
        if gate:
            ax.axhline(gate, color=RED, lw=1.2, label="0.90 gate")
        ax.set_xscale("log")
        ax.set_xticks([9, 18, 27, 54, 108])
        ax.set_xticklabels(["9", "18", "27", "54", "108"])
        ax.set_xlabel("reviewed days used to fit the intercept")
        ax.set_ylabel("held-out energy precision (median, 5–95% over draws)")
        src, tgt = direction.split("->")
        ax.set_title(f"slope from {src.capitalize()}, intercept from {tgt.capitalize()} reviewed days", fontsize=10.5)
        ax.grid(axis="y", color="#ebe3e3")
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    axes[0].legend(fontsize=8, loc="lower right")
    fig.suptitle("M9 intercept transfer: the scorer and the slope are frozen; only the intercept is fitted on the target population", fontsize=11)
    fig.tight_layout()
    fig.savefig(HERE / "fig_learning_curve.png", dpi=200, bbox_inches="tight")
    print("wrote fig_learning_curve.png")


if __name__ == "__main__":
    main()
