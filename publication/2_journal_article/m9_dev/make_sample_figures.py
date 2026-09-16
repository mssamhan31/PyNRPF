"""Twenty-seven site-day sample figures for the frozen M9 method.

Purpose: show what M9 does on real days, one panel per site-day, covering all four
decision outcomes against the reviewer's labels. Beta 'sure' days only, so every
panel is a real error or a real non-error that a reviewer was confident about.

Inputs:  runs/phase5_final/predictions_beta.csv; dataset/final/dataset_beta.parquet.
Outputs: figures/samples_TP.png (9), samples_FN.png (6), samples_FP.png (6),
         samples_TN.png (6), and figures/samples_index.csv listing every panel.
Selection (seeded, reproducible): TP = 3 largest corrected MWh + 3 closest to the
         threshold + 3 random; FN = 3 UNCERTAIN + 3 AUTO_KEEP by reference MWh;
         FP = the 6 largest applied MWh; TN = 3 closest to the threshold + 3 random.
Each panel: recorded net load, solar, underlying demand if the sign is kept
         (U0 = S + y), underlying demand if corrected (S - y inside the window),
         the proposed window and the labelled window.
"""

from __future__ import annotations

import pathlib

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = pathlib.Path(__file__).resolve().parent
RUN = HERE / "runs" / "phase5_final"
DATA = HERE.parent / "dataset" / "final" / "dataset_beta.parquet"
OUT = HERE / "figures"
C = 0.7
SEED = 9


def load_days() -> pd.DataFrame:
    df = pd.read_parquet(DATA)
    df["ts"] = pd.to_datetime(df["timestamp"], utc=True)
    df["d"] = df["ts"].dt.date.astype(str)
    return df.sort_values(["substation_id", "ts"])


def panel(ax, day: pd.DataFrame, row: pd.Series, kind: str) -> None:
    t = np.arange(96) / 4
    y = day["net_load_MW"].to_numpy(float)
    s = day["solar_MW"].to_numpy(float)
    lab = day["label_interval"].to_numpy(bool)
    u0 = s + y
    a, b = int(row["best_start"]), int(row["best_end"])
    applied = row["outcome"] == "AUTO_CORRECT"
    ax.plot(t, y, color="black", lw=1.3, label="recorded net load")
    ax.plot(t, s, color="tab:orange", lw=1.0, label="solar estimate")
    ax.plot(t, u0, color="grey", lw=1.0, ls="--", label="underlying load, sign kept")
    if a >= 0:
        uc = u0.copy()
        uc[a : b + 1] = s[a : b + 1] - y[a : b + 1]
        ax.plot(t, uc, color="tab:blue", lw=1.2, label="underlying load, corrected")
        ax.axvspan(a / 4, (b + 1) / 4, color="tab:blue" if applied else "tab:purple", alpha=0.15,
                   label="window applied" if applied else "window proposed, not applied")
    idx = np.flatnonzero(lab)
    if idx.size:
        ax.axvspan(idx[0] / 4, (idx[-1] + 1) / 4, color="tab:red", alpha=0.10, label="reviewer-labelled RPF")
    ax.axhline(0, color="k", lw=0.5)
    ax.set_xlim(0, 24)
    ax.set_xticks(range(0, 25, 6))
    ax.set_ylabel("MW", fontsize=8)
    ax.tick_params(labelsize=7)
    r = row["r_best"] if pd.notna(row["r_best"]) else float("nan")
    p = row["p"] if pd.notna(row["p"]) else float("nan")
    ax.set_title(f"{row['station']}  {row['date']}   {row['outcome']}\n"
                 f"p = {p:.2f}   r = {r:.1f}   reference {row['required_mwh']:.1f} MWh   proposed {row['proposed_mwh']:.1f} MWh",
                 fontsize=8)


def grid(rows: pd.DataFrame, kind: str, days: pd.DataFrame, name: str, ncol: int = 3) -> None:
    n = len(rows)
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(6.2 * ncol, 2.9 * nrow), squeeze=False)
    for k, (_, row) in enumerate(rows.iterrows()):
        ax = axes[k // ncol, k % ncol]
        day = days[(days["substation_id"] == row["station"]) & (days["d"] == row["date"])]
        panel(ax, day, row, kind)
    for k in range(n, nrow * ncol):
        axes[k // ncol, k % ncol].axis("off")
    h, lbl = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, lbl, loc="lower center", ncol=6, fontsize=8, frameon=False)
    fig.suptitle(f"M9 frozen method — {kind} samples, Beta 'sure' days, c = {C}", fontsize=11)
    fig.tight_layout(rect=(0, 0.04, 1, 0.97))
    fig.savefig(OUT / f"samples_{name}.png", dpi=120)
    plt.close(fig)


def main() -> None:
    OUT.mkdir(exist_ok=True)
    rng = np.random.default_rng(SEED)
    p = pd.read_csv(RUN / "predictions_beta.csv")
    p = p[(p["confidence"] == "sure") & p["input_ok"]].copy()
    p["dist"] = (p["p"] - C).abs()
    corr = p["outcome"] == "AUTO_CORRECT"
    tp, fn = p[corr & (p["rpf"] == 1)], p[~corr & (p["rpf"] == 1)]
    fp, tn = p[corr & (p["rpf"] == 0)], p[~corr & (p["rpf"] == 0)]

    def sample(df: pd.DataFrame, n: int) -> pd.DataFrame:
        return df.iloc[rng.choice(len(df), size=min(n, len(df)), replace=False)]

    tp_sel = pd.concat([tp.nlargest(3, "proposed_mwh"), tp.nsmallest(3, "dist"),
                        sample(tp.drop(tp.nlargest(3, "proposed_mwh").index).drop(tp.nsmallest(3, "dist").index, errors="ignore"), 3)])
    fn_sel = pd.concat([fn[fn["outcome"] == "UNCERTAIN"].nlargest(3, "required_mwh"),
                        fn[fn["outcome"] == "AUTO_KEEP"].nlargest(3, "required_mwh")])
    fp_sel = fp.nlargest(6, "proposed_mwh")
    tn_sel = pd.concat([tn.nsmallest(3, "dist"), sample(tn.drop(tn.nsmallest(3, "dist").index), 3)])

    days = load_days()
    grid(tp_sel, "TP — corrected, reviewer agrees", days, "TP")
    grid(fn_sel, "FN — not corrected, reviewer says RPF", days, "FN")
    grid(fp_sel, "FP — corrected, reviewer says no RPF", days, "FP")
    grid(tn_sel, "TN — kept, reviewer agrees", days, "TN")
    index = pd.concat([tp_sel.assign(kind="TP"), fn_sel.assign(kind="FN"), fp_sel.assign(kind="FP"), tn_sel.assign(kind="TN")])
    index[["kind", "station", "date", "outcome", "p", "r_best", "best_start", "best_end", "true_start", "true_end",
           "required_mwh", "proposed_mwh", "correct_mwh"]].to_csv(OUT / "samples_index.csv", index=False)
    print(f"{len(index)} panels: TP {len(tp_sel)}, FN {len(fn_sel)}, FP {len(fp_sel)}, TN {len(tn_sel)} -> {OUT}")


if __name__ == "__main__":
    main()
