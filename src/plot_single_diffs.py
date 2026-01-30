import pandas as pd
import proplot as pro
import numpy as np

import paths


def plot_table(table, filt_name, lp=True):
    table = table.sort_values(["hwp", "imr"])

    groups = table.groupby("imr")

    cmap = pro.Colormap("inferno", left=0.2, right=0.8, discrete=True)
    colors = cmap(np.linspace(0, 1, len(table["imr"].unique())))

    fig, axes = pro.subplots(width="4in", height=f"{4 / 1.1}in")

    for idx, (imr_ang, group) in enumerate(groups):
        axes.plot(
            group["hwp"],
            group["single_norm"],
            # bardata=group["single_norm_err"],
            c=colors[idx],
            marker="o",
            label=f"{imr_ang:.02f}°",
            zorder=1000,
        )

    axes.axhline(0, lw=1, c="0.2", zorder=100)
    axes.legend(ncols=3, loc="bottom", title="IMR angle")

    title = f"2025/11/26 MagAO-X LP=0 {filt_name}-band"
    if not lp:
        title = title.replace("LP=0", "No LP")
    axes.format(
        xlabel="HWP angle (°)",
        ylabel="Normalized single difference",
        title=title,
        ylim=(-1, 1) if lp else (-0.5, 0.5),
        xlocator=22.5,
    )

    fig.savefig(
        paths.figures / f"20251126_magaox_{'lp0' if lp else 'nolp'}_{filt_name}.pdf"
    )
    fig.savefig(
        paths.figures / f"20251126_magaox_{'lp0' if lp else 'nolp'}_{filt_name}.png"
    )

    pro.close()


if __name__ == "__main__":
    table = pd.read_csv(paths.data / "20251126_magaox_lp0_single_diffs.csv")

    for filt_name, group in table.groupby("filter"):
        plot_table(group, filt_name, lp=True)

    table = pd.read_csv(paths.data / "20251126_magaox_nolp_single_diffs.csv")

    for filt_name, group in table.groupby("filter"):
        plot_table(group, filt_name, lp=False)
