import pandas as pd
import ultraplot as pro
import numpy as np

import paths


def plot_table(table, filt_name, lp=True):
    table = table.sort_values(["hwp_ang", "imr_ang"])

    groups = table.groupby("hwp_ang")

    cmap = pro.Colormap("inferno", left=0.2, right=0.8, discrete=True)
    colors = cmap(np.linspace(0, 1, len(table["hwp_ang"].unique())))

    fig, axes = pro.subplots(width="4in", height=f"{4 / 1.1}in")

    for idx, (hwp_pair, group) in enumerate(groups):
        axes.plot(
            group["imr_ang"],
            group["double_diff"],
            # bardata=group["double_norm_err"],
            c=colors[idx],
            marker="o",
            label=hwp_pair,
            zorder=1000,
        )

    axes.axhline(0, lw=1, c="0.2", zorder=100)
    axes.legend(ncols=3, loc="bottom")

    title = f"2026/03/21 MagAO-X LP=0 {filt_name}-band"
    if not lp:
        title = title.replace("LP=0", "No LP")
    axes.format(
        xlabel="IMR angle (°)",
        ylabel="Normalized double difference",
        title=title,
        ylim=(-1, 1) if lp else (-0.5, 0.5),
    )

    fig.savefig(
        paths.figures
        / f"20260321_magaox_{'lp0' if lp else 'nolp'}_{filt_name}_double_diff.pdf"
    )
    fig.savefig(
        paths.figures
        / f"20260321_magaox_{'lp0' if lp else 'nolp'}_{filt_name}_double_diff.png"
    )

    pro.close()


if __name__ == "__main__":
    table_r = pd.read_csv(paths.data / "20260321_pdi_calib_lpV_double-diff_r.csv")
    table_i = pd.read_csv(paths.data / "20260321_pdi_calib_lpV_double-diff_i.csv")
    table_z = pd.read_csv(paths.data / "20260321_pdi_calib_lpV_double-diff_z.csv")

    plot_table(table_r, "r", lp=True)
    plot_table(table_i, "i", lp=True)
    plot_table(table_z, "z", lp=True)

    table_r = pd.read_csv(paths.data / "20260321_pdi_calib_nolp_double-diff_r.csv")
    table_i = pd.read_csv(paths.data / "20260321_pdi_calib_nolp_double-diff_i.csv")
    table_z = pd.read_csv(paths.data / "20260321_pdi_calib_nolp_double-diff_z.csv")

    plot_table(table_r, "r", lp=False)
    plot_table(table_i, "i", lp=False)
    plot_table(table_z, "z", lp=False)

