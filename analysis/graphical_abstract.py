"""Draw a graphical abstract from the representative full-model simulation."""

from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

def main():
    HERE = Path(__file__).resolve().parents[1] / "results" / "supplementary"
    data = pd.read_csv(HERE / "full_model_electrode_trajectories.csv")
    t = data.time_s / 60
    blue, orange = "#1768AC", "#D85A24"

    plt.rcParams.update({"font.family": "STIXGeneral", "mathtext.fontset": "stix",
                         "font.size": 11, "pdf.fonttype": 42, "ps.fonttype": 42})
    fig, (a, b) = plt.subplots(1, 2, figsize=(10.2, 3.45),
                             gridspec_kw={"wspace": 0.29})
    fig.patch.set_facecolor("white")

    for ax in (a, b):
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(labelsize=10, length=3)
        ax.set_xlim(0, 19)
        ax.axvline(t.iloc[-1], color="0.65", lw=1, ls=":")

    a.plot(t, data.xp_volume_mean_1, lw=2.6, color=blue, label="10% LLI")
    a.plot(t, data.xp_volume_mean_2, lw=2.6, color=orange,
           label="22.94% LAM$_n$")
    a.set(xlabel="Charge time (min)", ylabel="Mean LFP lithiation",
          ylim=(-0.025, 0.44))
    a.legend(loc="upper right", frameon=False, fontsize=10)
    a.text(0.01, 1.12, "Equal C/20 capacity", transform=a.transAxes,
           fontsize=14, fontweight="bold", ha="left")
    a.text(0.01, 1.025, "Different electrode operating windows",
           transform=a.transAxes, fontsize=10.5, ha="left")

    b.plot(t, data.I1, lw=2.6, color=blue, label="10% LLI")
    b.plot(t, data.I2, lw=2.6, color=orange, label="22.94% LAM$_n$")
    b.set(xlabel="Charge time (min)", ylabel="Branch current (A m$^{-2}$)",
          ylim=(-0.8, 27))
    b.axhline(12.41367, color="0.72", lw=1, ls="--")
    b.legend(loc="upper left", frameon=False, fontsize=10)
    b.text(0.01, 1.12, "Common terminal voltage", transform=b.transAxes,
           fontsize=14, fontweight="bold", ha="left")
    b.text(0.01, 1.025, "Transient handover near charge cutoff",
           transform=b.transAxes, fontsize=10.5, ha="left")

    fig.text(0.5, 0.008,
             "Electrode-state alignment, rather than capacity alone, governs transient current sharing.",
             ha="center", va="bottom", fontsize=12.2, color="#263238")
    fig.subplots_adjust(left=0.10, right=0.985, bottom=0.23, top=0.79)
    fig.savefig(HERE / "Graphical_Abstract_revised.pdf")
    fig.savefig(HERE / "Graphical_Abstract_revised.png", dpi=240)
    plt.close(fig)


if __name__ == "__main__":
    main()
