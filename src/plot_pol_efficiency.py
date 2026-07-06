import pandas as pd
import ultraplot as pro
import numpy as np

import paths
import plot_utils


def plot_table(table, filt_name, lp=True):
    table = table.sort_values(["hwp_ang", "imr_ang"])

    groups = table.groupby("imr_ang")

    cmap = pro.Colormap("inferno", left=0.2, right=0.8, discrete=True)
    colors = cmap(np.linspace(0, 1, len(table["imr_ang"].unique())))

    fig, axes = pro.subplots(width="4in", height=f"{4 / 1.1}in")

    for idx, (imr_ang, group) in enumerate(groups):
        axes.plot(
            group["hwp_ang"],
            group["single_diff"],
            # bardata=group["single_diff_err"],
            c=colors[idx],
            marker="o",
            label=f"{imr_ang:.02f}°",
            zorder=1000,
        )

    axes.axhline(0, lw=1, c="0.2", zorder=100)
    axes.legend(ncols=3, loc="bottom", title="IMR angle")

    title = f"2026/03/21 MagAO-X LP=0° {filt_name}-band"
    if not lp:
        title = title.replace("LP=0", "No LP")
    axes.format(
        xlabel="HWP angle (°)",
        ylabel="Normalized single difference",
        title=title,
        ylim=(-1, 1) if lp else (-0.5, 0.5),
        xlocator=22.5,
    )

    plot_utils.save_figure(
        fig, paths.figures / f"20260321_magaox_{'lp0' if lp else 'nolp'}_{filt_name}"
    )

    pro.close()


def get_pol_efficiency(group):
    return (group["single_diff"].max() - group["single_diff"].min()) / 2


if __name__ == "__main__":
    table_r = pd.read_csv(paths.data / "20260321_pdi_calib_lpV_r.csv")
    table_i = pd.read_csv(paths.data / "20260321_pdi_calib_lpV_i.csv")
    table_z = pd.read_csv(paths.data / "20260321_pdi_calib_lpV_z.csv")
    table_r["filter"] = "r"
    table_i["filter"] = "i"
    table_z["filter"] = "z"
    table = pd.concat((table_r, table_i, table_z))

    table["filter"] = pd.Categorical(table["filter"], ["r", "i", "z"])
    results = table.groupby(["filter", "imr_ang"]).apply(get_pol_efficiency)

    cmap = pro.Colormap("inferno", left=0.3, right=0.7, reverse=True, discrete=True)
    colors = cmap(np.linspace(0, 1, 3))
    color_dict = {"r": colors[0], "i": colors[1], "z": colors[2]}

    fig, axes = pro.subplots(width="4in", height=f"{4 / 1.4}in")
    for filt_name, group in results.groupby("filter"):
        imr_angs = np.array([idx[1] for idx in group.index])
        mask = (imr_angs <= -58.5) & (imr_angs >= -58.5 - 30)

        print(f"{filt_name}: {group.iloc[mask].mean() * 100:.01f}%")
        axes.plot(
            imr_angs,
            group.values * 100,
            c=color_dict[filt_name],
            marker="o",
            label=filt_name,
        )

    axes.legend(ncols=1, title="Filter", loc="ur", zorder=500)

    title = "2026/03/21 MagAO-X LP=0°"
    axes.format(
        xlabel="IMR angle (°)",
        ylabel="Est. polarimetric efficiency",
        title=title,
        yformatter="percent",
        ylim=(None, 100),
        xlocator=22.5,
    )
    axes.fill_between(
        [-58.5 - 30, -58.5], *axes.get_ylim(), lw=1, c="0.2", alpha=0.2, ec=None
    )

    plot_utils.save_figure(fig, paths.figures / "20260321_magaox_lp0_pol_eff")

    pro.close()
