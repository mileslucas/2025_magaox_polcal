import numpy as np
import plot_utils
import proplot as pro
import paths
import pandas as pd
import mueller_matrices as mm
from fit_model_A import model

def analyze_mm(X, imr_deg):
    S_in = np.array((1, 0, 0, 0))
    mmQp = model(X, 0, imr_deg, S_in=S_in)
    mmQm = model(X, 45, imr_deg, S_in=S_in)
    aveQ = 0.5 * (mmQp + mmQm)
    mmUp = model(X, 22.5, imr_deg, S_in=S_in)
    mmUm = model(X, 67.5, imr_deg, S_in=S_in)
    aveU = 0.5 * (mmUp + mmUm)
    avePI = np.hypot(aveQ, aveU) / np.sqrt(2)
    aveAoLP = 0.5 * np.arctan2(aveU, aveQ)
    return avePI

if __name__ == "__main__":

    X_r = [
        0.12763846035969215,
        0.4719104582373841,
        0.1673610501405619, # m4 diatt
        0.5860550336830435,
        0.11308308928246653, 
        0.019031317607203783,# imr diatt
        0.3725305893993576,
        0.28197682200646423,
        0.5206500345992562,
        0.1708904521683427, 
        -0.48071908331945395,# inst diatt
        0.9999994711696236,
        8.782425975252154e-07,
        0.6488984072036339,
        0.22282390293724305,
        0.9999889527837509,
    ]

    X_i = [
        -0.006312809798119445,
        0.5262250073378643,
        0.07435245442238711, # m4 diatt
        0.5725899415530027,
        -0.04174720114543014,
        -0.060518086641387255,# imr diatt
        0.19905771029080493,
        0.3501505051770073,
        0.40729196209672297,
        0.06782440682377217,
        -0.006099828740533294,# inst diatt
        0.7731846388858752,
        0.2405402946867487,
        0.9994227916301208,
        0.9343181089660519,
        0.9329688990441918,
    ]
    X_z = [
        -0.1194854244655808,
        0.5374174438910502,
        0.12798154409744494, # m4 diatt
        0.5478678986031353,
        0.040835920644347705,
        -0.03665345375744013,# imr diatt
        0.6080012064975125,
        0.15674883502671277,
        0.280198252550408,
        0.07129555241731772,
        -0.06839805228790993,# inst diatt
        0.44109385844200666,
        0.46899652606746545,
        0.636779435298549,
        0.9997003639129332,
        0.9842176030536884,
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
            zorder=500,
            label=filt_name
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

    axes.legend(ncols=1, title="Filter", loc="ur", zorder=500)

    title = "2025/11/26 MagAO-X No LP"
    axes.format(
        xlabel="IMR angle (°)",
        ylabel="Instrumental polarization",
        title=title,
        yformatter="percent",
        ylim=(0, 100),
        xlocator=22.5,
    )
    axes.fill_between(
        [-58.5 - 30, -58.5], *axes.get_ylim(), lw=1, c="0.2", alpha=0.2, ec=None
    )

    plot_utils.save_figure(fig, paths.figures / "20251125_magaox_model_inst_pol")

    df = pd.DataFrame(rows)

    # df.to_csv(paths.data / "pol_eff_model_report.csv", index=False)
    print(df)
