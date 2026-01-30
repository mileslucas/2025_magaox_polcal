import pandas as pd
import proplot as pro
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
    table = table.sort_values(["hwp", "imr"])

    groups = table.groupby("hwp")

    cmap = pro.Colormap("inferno", left=0.2, right=0.8, discrete=True)
    colors = cmap(np.linspace(0, 1, len(table["hwp"].unique())))

    fig, axes = pro.subplots(
        nrows=2, width="4in", height=f"{4 / 1.05}in", height_ratios=(2, 1), sharey=False
    )

    mse = 0
    for idx, (hwp_str, group) in enumerate(groups):
        axes[0].scatter(
            group["imr"].values,
            group["double_norm"].values,
            # bardata=group["double_norm_err"],
            c=colors[idx],
            marker="o",
            label=f"{hwp_str}",
            zorder=1000,
        )
        test_imr = np.linspace(group["imr"].min(), group["imr"].max(), 100)
        if samples is not None:
            curve, curve_std = get_model_curve(samples, hwp_str, group["imr"])
            test_curve, test_std = get_model_curve(samples, hwp_str, test_imr)
        else:
            if filt_name == "r":
                X = [
                    -0.009476013790448247,
                    0.4600570199618508,
                    0.5809054504835991,
                    0.0003840381726163845,
                    0.7664635139372711,
                    0.06868191785095659,
                    0.24486261134275322,
                    -0.0034989763086488703,
                    8.232615066370506e-05,
                    0.4108894437539104,
                    0.7962058116235509,
                    0.7377096762073958,
                ]

            elif filt_name == "i":
                X = [
                    -0.0032664002024398514,
                    0.4643883673835176,
                    0.5977157796493537,
                    0.0025640438892886993,
                    0.6862109309309716,
                    0.18617537190994693,
                    0.2847442035223146,
                    -0.004263712374693801,
                    0.5497421259867903,
                    0.26720659407740677,
                    0.6815646684652745,
                    0.944406385043694,
                ]
            elif filt_name == "z":
                X = [
                    0.0003257816304214747,
                    0.49493813939135,
                    0.5590224291792651,
                    0.0020298679467539355,
                    0.28115800304344984,
                    0.36859618523344684,
                    0.37509136820947164,
                    -0.00024504894314898685,
                    0.9993863660727822,
                    0.2875288549142562,
                    0.7111705511572872,
                    0.9747455202147504,
                ]
            S_in = np.array([1, 1, 0, 0])
            curve = np.array(
                [fit_model_B.model(X, hwp_str, i, S_in) for i in group["imr"]]
            )
            test_curve = np.array(
                [fit_model_B.model(X, hwp_str, i, S_in) for i in test_imr]
            )

        resids = group["double_norm"].values - curve
        mse += np.nanmean(resids**2) / len(groups)

        hwp_str_tokens = hwp_str.split("-")
        label = f"{hwp_str_tokens[0]}°-{hwp_str_tokens[1]}°"
        axes[1].scatter(
            group["imr"].values, resids, c=colors[idx], zorder=999, label=label
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
        f"MSE={mse * 100:.01f}%",
        fontsize=8,
        color="0.2",
        ha="right",
        va="bottom",
        transform="axes",
    )
    axes[0].axhline(0, lw=1, c="0.2", zorder=100)
    axes[1].axhline(0, lw=1, c="0.2", zorder=100)

    axes[1].legend(ncols=3, loc="bottom", title="HWP pair", fontsize=8)

    title = f"2025/11/26 MagAO-X LP=0° {filt_name}-band"
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
        paths.figures / f"20251126_magaox_{'lp0' if lp else 'nolp'}_{filt_name}_model",
    )

    pro.close()


if __name__ == "__main__":
    table = pd.read_csv(paths.data / "20251126_magaox_lp0_double_diffs.csv")

    for filt_name, group in table.groupby("filter"):
        try:
            raise KeyError()
            npz_data = np.load(paths.data / f"model_B_chains_{filt_name}.npz")
            plot_table(group, filt_name, samples=npz_data["samples"], lp=True)
        except Exception:
            plot_table(group, filt_name, samples=None, lp=True)

    # table = pd.read_csv(paths.data / "20251126_magaox_nolp_single_diffs.csv")

    # for filt_name, group in table.groupby("filter"):
    #     plot_table(group, filt_name, lp=False)
