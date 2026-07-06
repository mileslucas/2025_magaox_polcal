from fit_model_B import model
import numpy as np
import plot_utils
import ultraplot as pro
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
        -0.763357229468489,
        0.52034088640206,
        0.9824289961809236,
        1.4273204717931574,
        0.03864471256555264,
        0.009426001176574453,
        2.1665579344132626e-09,
        0.12971439446151686,
        0.5418446560784482,
        4.7715159580802585e-06,
        0.02265987009820017,
        0.6905410362332749,
    ]

    X_i = [
        0.0010561204605654788,
        0.5399999789977414,
        0.7228761048426008,
        0.0010627376194343595,
        0.0014282256396114452,
        0.016386897914529536,
        0.019921780572015463,
        -0.004120654828559514,
        0.014036445847131037,
        0.01523616188516444,
        0.04106393440179849,
        0.9999978102225718,
    ]
    X_z = [
        -0.01219543133540035,
        0.5399508525462158,
        0.7790335597326843,
        -0.007179981683667631,
        0.020326480196856178,
        0.022536487016844,
        0.037284012265062325,
        -0.012079293124252436,
        0.04462150550670938,
        2.2654876313657055e-08,
        0.029728954288789782,
        0.999848042512433,
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
        # axes.axvline(
        #     abs_peak_imrang, lw=1, ls="--", c=color_dict[filt_name], zorder=300
        # )
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

    title = "2026/03/21 MagAO-X LP=0°"
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

    plot_utils.save_figure(fig, paths.figures / "20260321_magaox_model_pol_eff")

    df = pd.DataFrame(rows)

    df.to_csv(paths.data / "pol_eff_model_report.csv", index=False)
    print(df)
