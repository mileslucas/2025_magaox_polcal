import pandas as pd
import proplot as pro
import numpy as np

import paths
import plot_utils
import fit_model_A_constrained


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
        curve = np.array(
            [fit_model_A_constrained.model(X, h, imr_ang, S_in) for h in hwp_angs]
        )
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
                    -0.10277930592375153,
                    -0.11685655079864947,
                    -0.5570911387907846,
                    0.22605929268044472,
                    0.9999988747248703,
                ]
                X2 = fit_model_A_constrained.X_r

            elif filt_name == "i":
                X = [
                    -0.15818557643327583,
                    0.07773938589239711,
                    -0.21101997552834859,
                    0.9538926409134032,
                    0.93284727677225,
                ]
                X2 = fit_model_A_constrained.X_i
            elif filt_name == "z":
                X = [
                    -0.035511325437657434,
                    -0.014793813629596827,
                    -0.199639571016783,
                    1.0,
                    0.9821458738822328,
                ]
                X2 = fit_model_A_constrained.X_z
            curve = np.array(
                [
                    fit_model_A_constrained.model(X, X2, h, imr_ang, S_in)
                    for h in group["hwp"]
                ]
            )
            test_curve = np.array(
                [
                    fit_model_A_constrained.model(X, X2, h, imr_ang, S_in)
                    for h in test_hwp
                ]
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
            if filt_name == "r":
                X2 = fit_model_A_constrained.X_r
            elif filt_name == "i":
                X2 = fit_model_A_constrained.X_i
            elif filt_name == "z":
                X2 = fit_model_A_constrained.X_z
            curve = np.array(
                [fit_model_A_constrained.model(X, X2, h, imr_ang, S_in) for h in group["hwp"]]
            )
            test_curve = np.array(
                [fit_model_A_constrained.model(X, X2, h, imr_ang, S_in) for h in test_hwp]
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
        / f"20251126_magaox_{'lp0' if lp else 'nolp'}_{filt_name}_single_diff_model_constrained",
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
