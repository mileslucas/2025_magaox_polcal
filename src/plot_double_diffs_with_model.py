import pandas as pd
import ultraplot as pro
import numpy as np

import paths
import plot_utils
import fit_model_B


def get_model_curve(samples, hwp_str, imr_ang: float, N=1000):
    curves = []
    X_samples = np.array(
        [
            np.random.choice(samples[:, j], size=N, replace=True)
            for j in range(samples.shape[1])
        ]
    )
    for i in range(N):
        X = X_samples[:, i]
        S_in = np.array([1, 1, 0, 0])
        curve = np.array([fit_model_B.model(X, hwp_str, i, S_in) for i in imr_ang])
        curves.append(curve)
    return np.mean(curves, 0), np.std(curves, 0)


def plot_table(table, filt_name, samples, lp=True):
    table = table.sort_values(["hwp_ang", "imr_ang"])

    groups = table.groupby("hwp_ang")

    cmap = pro.Colormap("inferno", left=0.2, right=0.8, discrete=True)
    colors = cmap(np.linspace(0, 1, len(table["hwp_ang"].unique())))

    fig, axes = pro.subplots(
        nrows=2, width="4in", height=f"{4 / 1.05}in", height_ratios=(2, 1), sharey=False
    )

    mse = 0
    for idx, (hwp_str, group) in enumerate(groups):
        axes[0].scatter(
            group["imr_ang"].values,
            group["double_diff"].values,
            # bardata=group["double_diff_err"],
            c=colors[idx],
            marker="o",
            label=f"{hwp_str}",
            zorder=1000,
        )
        test_imr = np.linspace(group["imr_ang"].min(), group["imr_ang"].max(), 100)
        if samples is not None:
            curve, curve_std = get_model_curve(samples, hwp_str, group["imr_ang"])
            test_curve, test_std = get_model_curve(samples, hwp_str, test_imr)
        else:
            if filt_name == "r":
                X = [
-0.09197337909586643,
0.5190747298697496,
0.8065014300951933,
0.1877031118489375,
0.0,
0.0021476830130610198,
0.037478361091875426,
0.4639795100540668,
0.02060977780460021,
8.791065290913073e-06,
5.958098846355867e-06,
-0.4634550959338772,
0.006517973123873337,
0.0004185212057480607,
0.00013622005166947446,
-0.13445991494969717,
0.0806267106308508,
9.603871140236828e-05,
0.1054756750812547,
0.999879639085673,
                ]

            elif filt_name == "i":
                X = [
-0.0028521364309477197,
0.539908170676348,
0.7436338744359774,
-0.00018695943606796185,
1.4734528970995503e-05,
0.0058569661159633074,
0.008856627653649335,
0.003719543302630277,
0.03262544185756816,
0.002631050859523087,
0.017941712942315148,
0.005168978128238624,
0.0034152312165470758,
0.010178489549924478,
0.03247753222378161,
0.0030182743512139967,
0.0026476742022171445,
0.01177622075454878,
0.001954271916517009,
1.0,
                ]
            elif filt_name == "z":
                X = [
0.0004594422984608936,
0.5399840430098184,
0.7742850960603289,
1.0472965571277279e-05,
0.01007936760455377,
0.02108081596919878,
0.025680846096936376,
0.017639289740441447,
0.009717940261215665,
1.5161586638046216e-05,
0.01871785394584312,
0.012264937037726732,
0.00968149441608107,
0.0,
0.006703992824105897,
-0.0010610662819890912,
0.0002660676668102255,
0.0,
0.016905875373672795,
0.9999387187296811,
                ]
            S_in = np.array([1, 1, 0, 0])
            curve = np.array(
                [fit_model_B.model(X, hwp_str, i, S_in) for i in group["imr_ang"]]
            )
            test_curve = np.array(
                [fit_model_B.model(X, hwp_str, i, S_in) for i in test_imr]
            )

        resids = group["double_diff"].values - curve
        mse += np.nanmean(resids**2) / len(groups)

        hwp_str_tokens = hwp_str.split("-")
        label = f"{hwp_str_tokens[0]}°-{hwp_str_tokens[1]}°"
        axes[1].scatter(
            group["imr_ang"].values, resids, c=colors[idx], zorder=999, label=label
        )

        axes[0].plot(
            test_imr,
            test_curve,
            # shadedata=test_std if samples is not None else None,
            c=colors[idx],
            zorder=999,
        )
    axes[1].text(
        0.97,
        1.02,
        f"MSE={mse * 100:.02f}%",
        fontsize=8,
        color="0.2",
        ha="right",
        va="bottom",
        transform="axes",
    )
    axes[0].axhline(0, lw=1, c="0.2", zorder=100)
    axes[1].axhline(0, lw=1, c="0.2", zorder=100)

    axes[1].legend(ncols=3, loc="bottom", title="HWP pair", fontsize=8)

    title = f"2026/03/21 MagAO-X LP=0° {filt_name}-band"
    if not lp:
        title = title.replace("LP=0°", "No LP")
    axes[0].format(
        ylabel="Norm. double difference",
        title=title,
        ylim=(-1, 1),
        xlocator=22.5,
    )
    ax1_ylim = max(*np.abs(axes[1].get_ylim()))
    axes[1].format(
        ylabel="Residual",
        xlabel="IMR angle (°)",
        xlocator=22.5,
        ylim=(-ax1_ylim, ax1_ylim),
    )

    plot_utils.save_figure(
        fig,
        paths.figures / f"20260321_magaox_{'lp0' if lp else 'nolp'}_{filt_name}_model",
    )

    pro.close()


if __name__ == "__main__":
    table_r = pd.read_csv(paths.data / "20260321_pdi_calib_lpV_double-diff_r.csv")
    table_i = pd.read_csv(paths.data / "20260321_pdi_calib_lpV_double-diff_i.csv")
    table_z = pd.read_csv(paths.data / "20260321_pdi_calib_lpV_double-diff_z.csv")

    plot_table(table_r, "r", samples=None, lp=True)
    plot_table(table_i, "i", samples=None, lp=True)
    plot_table(table_z, "z", samples=None, lp=True)
