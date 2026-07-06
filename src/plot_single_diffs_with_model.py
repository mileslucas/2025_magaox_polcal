import pandas as pd
import ultraplot as pro
import numpy as np

import paths
import plot_utils
import fit_model_A


def get_model_curve(samples, hwp_angs, imr_ang: float, S_in, N=1000):
    curves = []
    X_samples = np.array(
        [
            np.random.choice(samples[:, j], size=N, replace=True)
            for j in range(samples.shape[1])
        ]
    )
    for i in range(N):
        X = X_samples[:, i]
        curve = np.array([fit_model_A.model(X, h, imr_ang, S_in) for h in hwp_angs])
        curves.append(curve)
    return np.mean(curves, 0), np.std(curves, 0)


def plot_tables(table, table_nolp, filt_name, samples, lp=True):
    table = table.sort_values(["hwp", "imr"])
    table_nolp = table_nolp.sort_values(["hwp", "imr"])

    groups = table.groupby("imr")
    groups_nolp = table_nolp.groupby("imr")

    cmap = pro.Colormap("inferno", left=0.2, right=0.8, discrete=True)
    colors = cmap(np.linspace(0, 1, len(table["imr"].unique())))
    colors_dict = dict(zip(table["imr"].unique(), colors))

    fig, axes = pro.subplots(
        nrows=2,
        ncols=2,
        width="7.5in",
        height=f"{4 / 1.1}in",
        height_ratios=(2, 1),
        spany=False,
    )

    S_in = np.array([1, 1, 0, 0])

    mse = 0
    for imr_ang, group in groups:
        axes[0, 0].scatter(
            group["hwp"],
            group["single_norm"],
            # bardata=group["single_norm_err"],
            c=colors_dict[imr_ang],
            marker="o",
            label=f"{imr_ang:.02f}°",
            zorder=1000,
        )
        test_hwp = np.linspace(group["hwp"].min(), group["hwp"].max(), 100)
        if samples is not None:
            curve, curve_std = get_model_curve(samples, group["hwp"], imr_ang, S_in)
            test_curve, test_std = get_model_curve(samples, test_hwp, imr_ang, S_in)
        else:
            if filt_name == "r":
                X = [
                    0.12763846035969215,
                    0.4719104582373841,
                    0.1673610501405619,
                    0.5860550336830435,
                    0.11308308928246653,
                    0.019031317607203783,
                    0.3725305893993576,
                    0.28197682200646423,
                    0.5206500345992562,
                    0.1708904521683427,
                    -0.48071908331945395,
                    0.9999994711696236,
                    8.782425975252154e-07,
                    0.6488984072036339,
                    0.22282390293724305,
                    0.9999889527837509,
                ]

            elif filt_name == "i":
                X = [
                    -0.006312809798119445,
                    0.5262250073378643,
                    0.07435245442238711,
                    0.5725899415530027,
                    -0.04174720114543014,
                    -0.060518086641387255,
                    0.19905771029080493,
                    0.3501505051770073,
                    0.40729196209672297,
                    0.06782440682377217,
                    -0.006099828740533294,
                    0.7731846388858752,
                    0.2405402946867487,
                    0.9994227916301208,
                    0.9343181089660519,
                    0.9329688990441918,
                ]
            elif filt_name == "z":
                X = [
                    -0.1194854244655808,
                    0.5374174438910502,
                    0.12798154409744494,
                    0.5478678986031353,
                    0.040835920644347705,
                    -0.03665345375744013,
                    0.6080012064975125,
                    0.15674883502671277,
                    0.280198252550408,
                    0.07129555241731772,
                    -0.06839805228790993,
                    0.44109385844200666,
                    0.46899652606746545,
                    0.636779435298549,
                    0.9997003639129332,
                    0.9842176030536884,
                ]
            curve = np.array(
                [fit_model_A.model(X, h, imr_ang, S_in) for h in group["hwp"]]
            )
            test_curve = np.array(
                [fit_model_A.model(X, h, imr_ang, S_in) for h in test_hwp]
            )

        axes[0, 0].plot(test_hwp, test_curve, c=colors_dict[imr_ang], zorder=999)

        resids = group["single_norm"].values - curve
        mse += np.nanmean(resids**2) / len(groups)

        axes[1, 0].scatter(group["hwp"], resids, c=colors_dict[imr_ang], zorder=999)
    text_kwargs = dict(ha="right", va="bottom", c="0.2", fontsize=8, transform="axes")
    axes[1, 0].text(0.97, 1.02, f"MSE={mse * 100:0.01f}%", **text_kwargs)

    mse = 0
    for imr_ang, group in groups_nolp:
        axes[0, 1].scatter(
            group["hwp"],
            group["single_norm"],
            # bardata=group["single_norm_err"],
            c=colors_dict[imr_ang],
            marker="o",
            label="",
            zorder=1000,
        )
        test_hwp = np.linspace(group["hwp"].min(), group["hwp"].max(), 100)
        S_in = np.array([1, 0, 0, 0])
        if samples is not None:
            curve, curve_std = get_model_curve(samples, group["hwp"], imr_ang, S_in)
            test_curve, test_std = get_model_curve(samples, test_hwp, imr_ang, S_in)
        else:
            curve = np.array(
                [fit_model_A.model(X, h, imr_ang, S_in) for h in group["hwp"]]
            )
            test_curve = np.array(
                [fit_model_A.model(X, h, imr_ang, S_in) for h in test_hwp]
            )

        axes[0, 1].plot(test_hwp, test_curve, c=colors_dict[imr_ang], zorder=999)

        resids = group["single_norm"].values - curve
        mse += np.nanmean(resids**2) / len(groups)

        axes[1, 1].scatter(group["hwp"], resids, c=colors_dict[imr_ang], zorder=999)

    axes[1, 1].text(0.97, 1.02, f"MSE={mse * 100:0.01f}%", **text_kwargs)

    axes[0, 0].axhline(0, lw=1, c="0.2", zorder=100)
    axes[0, 1].axhline(0, lw=1, c="0.2", zorder=100)
    axes[1, 0].axhline(0, lw=1, c="0.2", zorder=100)
    axes[1, 1].axhline(0, lw=1, c="0.2", zorder=100)

    fig.legend(ncols=5, loc="top", title="IMR angle")

    title = f"2025/11/26 MagAO-X LP=0 {filt_name}-band"
    if not lp:
        title = title.replace("LP=0", "No LP")
    axes[0, 0].format(
        ylabel="Norm. single difference",
    )
    axes[0, :].format(
        # title=title,
        ylim=(-1, 1),
        xlocator=22.5,
    )
    axes[1, 0].format(
        ylabel="Residual",
    )
    axes[1, :].format(xlabel="HWP angle (°)", xlocator=22.5)
    plot_utils.save_figure(
        fig,
        paths.figures
        / f"20251126_magaox_{'lp0' if lp else 'nolp'}_{filt_name}_single_diff_model",
    )

    pro.close()


if __name__ == "__main__":
    table = pd.read_csv(paths.data / "20251126_magaox_lp0_single_diffs.csv")
    table_nolp = pd.read_csv(paths.data / "20251126_magaox_nolp_single_diffs.csv")

    for (filt_name, group), (_, group_nolp) in zip(
        table.groupby("filter"), table_nolp.groupby("filter")
    ):
        try:
            npz_data = np.load(paths.data / f"model_A_chains_{filt_name}.npz")
            # plot_tables(group, group_nolp, filt_name, samples=npz_data["samples"], lp=True)
        except Exception:
            ...
        plot_tables(group, group_nolp, filt_name, samples=None, lp=True)

    # for filt_name, group in table.groupby("filter"):
    #     plot_table(group, filt_name, lp=False)
