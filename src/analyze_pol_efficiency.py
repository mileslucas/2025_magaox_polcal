from fit_model_B import model
import numpy as np
import plot_utils
import proplot as pro
import paths
import pandas as pd


def calculate_pol_efficiency(mmQ, mmU):
    poleff_Q = np.hypot(mmQ[1], mmU[1])
    poleff_U = np.hypot(mmQ[2], mmU[2])
    poleff_QU = np.sqrt(0.5 * (mmQ[1] + mmQ[2]) ** 2 + 0.5 * (mmU[1] + mmU[2]) ** 2)
    average_poleff = np.mean((poleff_Q, poleff_U, poleff_QU))
    return average_poleff


def analyze_mm(X, imr_deg):
    mmQ = model(X, "0-45", imr_deg, S_in=None)[0]
    mmU = model(X, "22.5-67.5", imr_deg, S_in=None)[0]
    polarimetric_efficiency = calculate_pol_efficiency(mmQ, mmU)
    return polarimetric_efficiency


if __name__ == "__main__":
    plot_utils.setup_plots()

    X_r = [
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
    X_i = [
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
    X_z = [
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
    imr_angs = np.linspace(-90, 0, 1000)
    measured_imr_angs = np.linspace(-90, 0, 9)

    cmap = pro.Colormap("inferno", left=0.3, right=0.7, reverse=True, discrete=True)
    colors = cmap(np.linspace(0, 1, 3))
    color_dict = {"r": colors[0], "i": colors[1], "z": colors[2]}

    fig, axes = pro.subplots(width="4in", height=f"{4 / 1.4}in")
    rows = []
    for filt_name in ("r", "i", "z"):
        pol_effs = np.array(
            [analyze_mm(eval(f"X_{filt_name}"), imr_ang) for imr_ang in imr_angs]
        )
        axes.plot(imr_angs, pol_effs * 100, c=color_dict[filt_name], zorder=400)

        measured_pol_effs = np.array(
            [
                analyze_mm(eval(f"X_{filt_name}"), imr_ang)
                for imr_ang in measured_imr_angs
            ]
        )
        axes.scatter(
            measured_imr_angs,
            measured_pol_effs * 100,
            c=color_dict[filt_name],
            label=filt_name,
            zorder=500,
        )

        mask = (imr_angs <= -58.5) & (imr_angs >= -58.5 - 30)
        ave_val = pol_effs[mask].mean()
        ave_imr = imr_angs[mask].mean()
        # axes.hlines(ave_val * 100, -90, -58.5, colors=color_dict[filt_name], lw=1, alpha=0.6)
        # axes.scatter(-90, ave_val * 100, c=color_dict[filt_name], marker="<")

        abs_peak_idx = pol_effs.argmax()
        abs_peak_imrang = imr_angs[abs_peak_idx]
        abs_peak_poleff = pol_effs[abs_peak_idx]
        axes.axvline(
            abs_peak_imrang, lw=1, ls="--", c=color_dict[filt_name], zorder=300
        )
        # axes.scatter(
        #     abs_peak_imrang, abs_peak_poleff * 100, c=color_dict[filt_name], marker="^"
        # )

        peak_idx = pol_effs[mask].argmax()
        peak_imrang = imr_angs[mask][peak_idx]
        peak_poleff = pol_effs[mask][peak_idx]

        min_idx = pol_effs[mask].argmin()
        min_imrang = imr_angs[mask][min_idx]
        min_poleff = pol_effs[mask][min_idx]

        rows.append(
            {
                "filter": filt_name,
                "abs_peak_imr": abs_peak_imrang,
                "abs_peak_eff": abs_peak_poleff,
                "peak_imr": peak_imrang,
                "peak_eff": peak_poleff,
                "min_imr": min_imrang,
                "min_eff": min_poleff,
                "ave_imr": ave_imr,
                "ave_eff": ave_val,
            }
        )

    axes.legend(ncols=1, title="Filter", loc="lr", zorder=500)

    title = "2025/11/26 MagAO-X LP=0°"
    axes.format(
        xlabel="IMR angle (°)",
        ylabel="polarimetric efficiency",
        title=title,
        yformatter="percent",
        ylim=(0, 100),
        xlocator=22.5,
    )
    axes.fill_between(
        [-58.5 - 30, -58.5], *axes.get_ylim(), lw=1, c="0.2", alpha=0.2, ec=None
    )

    plot_utils.save_figure(fig, paths.figures / "20251125_magaox_model_pol_eff")

    df = pd.DataFrame(rows)

    df.to_csv(paths.data / "pol_eff_model_report.csv", index=False)
    print(df)
