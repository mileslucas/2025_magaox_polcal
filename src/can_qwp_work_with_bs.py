import numpy as np
import plot_utils
import proplot as pro
import paths
import pandas as pd
import mueller_matrices as mm
from scipy.optimize import minimize

X_r = [
    0.005748516561452863,
    0.5399938735450269,
    0.2657393445107225,
    0.583802620336048,
    65.16415822460831,
    0.07574812773324914,
    0.5086442058641332,
    0.5116765948225965,
    8.14961570913233,
    0.28261388048241287,
    # 0.7579220531638995,
    1 - 1/500,
]
X_i = [
    -0.0014926557833302863,
    0.5356417395993677,
    0.00013001807625580557,
    0.5977181376299912,
    98.13180378793746,
    0.5095019654974758,
    0.19326775014909767,
    0.4441314156213433,
    7.247016360152024,
    0.34238930576299553,
    # 0.9444371034054817,
    1 - 1/500,
]
X_z = [
    0.006129551664538023,
    0.5049754472379833,
    0.0385267934112775,
    0.5589831881802595,
    98.26669045690684,
    0.27299995043159975,
    0.3113088297022478,
    0.3633317114524698,
    19.705367842769927,
    0.4361473128726396,
    # 0.975356992146905,
    1 - 1/500,
]

qwp_design_wavelength = 760


def model(X, X2, hwp_str, imr_deg, S_in):
    """
    Model B is

    """
    (
        qwp1_deg,
        qwp2_deg,
        # qwp_ret
    ) = X
    (
        hwp_offset,
        hwp_retardance,
        m4_diatt,
        m4_retardance,
        imr_offset,
        imr_retardance_h,
        imr_retardance_45,
        imr_retardance_r,
        inst_offset,
        inst_retardance,
        pbs_extinction,
    ) = X2
    if X2 == X_r:
        qwp_ret = get_retardance_relative("r", qwp_design_wavelength)
    elif X2 == X_i:
        qwp_ret = get_retardance_relative("i", qwp_design_wavelength)
    elif X2 == X_z:
        qwp_ret = get_retardance_relative("z", qwp_design_wavelength)

    hwp_str1, hwp_str2 = hwp_str.split("-")
    hwp_deg1 = float(hwp_str1)
    hwp_deg2 = float(hwp_str2)
    hwp_theta1 = np.deg2rad(hwp_deg1 + hwp_offset)
    hwp_theta2 = np.deg2rad(hwp_deg2 + hwp_offset)
    imr_theta = np.deg2rad(imr_deg + imr_offset)

    qwp1_theta = np.deg2rad(qwp1_deg)
    qwp2_theta = np.deg2rad(qwp2_deg)

    hwp_diatt = 0
    mm_hwp1 = mm.generic(hwp_theta1, hwp_diatt, hwp_retardance * 2 * np.pi)
    mm_hwp2 = mm.generic(hwp_theta2, hwp_diatt, hwp_retardance * 2 * np.pi)
    # mm_hwp1 = mm.elliptical_retarder(
    #     hwp_theta1,
    #     np.deg2rad(hwp_retardance_h),
    #     np.deg2rad(hwp_retardance_45),
    #     np.deg2rad(hwp_retardance_r),
    # )
    # mm_hwp2 = mm.elliptical_retarder(
    #     hwp_theta2,
    #     np.deg2rad(hwp_retardance_h),
    #     np.deg2rad(hwp_retardance_45),
    #     np.deg2rad(hwp_retardance_r),
    # )

    mm_m4 = mm.generic(0, m4_diatt, m4_retardance * 2 * np.pi)

    # imr_diatt = 0
    # mm_imr = mm.generic(
    #     imr_theta,
    #     imr_diatt,
    #     imr_retardance * 2 * np.pi,
    # )
    mm_imr = mm.elliptical_retarder(
        imr_theta,
        imr_retardance_h * 2 * np.pi,
        imr_retardance_45 * 2 * np.pi,
        imr_retardance_r * 2 * np.pi,
    )

    mm_qwp1 = mm.generic(qwp1_theta, delta=qwp_ret * 2 * np.pi)
    mm_qwp2 = mm.generic(qwp2_theta, delta=qwp_ret * 2 * np.pi)

    inst_diatt = 0
    mm_inst = mm.generic(
        np.deg2rad(inst_offset), inst_diatt, inst_retardance * 2 * np.pi
    )
    # mm_inst = mm.elliptical_retarder(
    #     np.deg2rad(90 + inst_offset),
    #     np.deg2rad(inst_retardance_h),
    #     np.deg2rad(inst_retardance_45),
    #     np.deg2rad(inst_retardance_r),
    # )
    # mm_inst = np.eye(4)

    # pbs_Rp = 2 - pbs_Rs - pbs_Tp - pbs_Ts
    # mm_pbs_V = 0.5 * np.array([[pbs_Rs + pbs_Rp, pbs_Rs - pbs_Rp, 0, 0],
    #                           [pbs_Rs - pbs_Rp, pbs_Rs + pbs_Rp, 0, 0],
    #                           [0, 0, 2*np.sqrt(pbs_Rs * pbs_Rp), 0],
    #                           [0, 0, 0, 2*np.sqrt(pbs_Rs * pbs_Rp)]])
    # mm_pbs_H = 0.5 * np.array([[pbs_Ts + pbs_Tp, pbs_Ts - pbs_Tp, 0, 0],
    #                           [pbs_Ts - pbs_Tp, pbs_Ts + pbs_Tp, 0, 0],
    #                           [0, 0, 2*np.sqrt(pbs_Ts * pbs_Tp), 0],
    #                           [0, 0, 0, 2*np.sqrt(pbs_Ts * pbs_Tp)]])
    mm_pbs_V = mm.wollaston(ordinary=True, epsilon=np.abs(pbs_extinction))
    mm_pbs_H = mm.wollaston(ordinary=False, epsilon=np.abs(pbs_extinction))

    total_mm_V1 = mm_pbs_V @ mm_inst @ mm_qwp2 @ mm_qwp1 @ mm_imr @ mm_m4 @ mm_hwp1
    total_mm_H1 = mm_pbs_H @ mm_inst @ mm_qwp2 @ mm_qwp1 @ mm_imr @ mm_m4 @ mm_hwp1
    mm_diff1 = total_mm_V1 - total_mm_H1
    mm_sum1 = total_mm_V1 + total_mm_H1

    total_mm_V2 = mm_pbs_V @ mm_inst @ mm_qwp2 @ mm_qwp1 @ mm_imr @ mm_m4 @ mm_hwp2
    total_mm_H2 = mm_pbs_H @ mm_inst @ mm_qwp2 @ mm_qwp1 @ mm_imr @ mm_m4 @ mm_hwp2
    mm_diff2 = total_mm_V2 - total_mm_H2
    mm_sum2 = total_mm_V2 + total_mm_H2

    mm_diff = 0.5 * (mm_diff1 - mm_diff2)
    mm_sum = 0.5 * (mm_sum1 + mm_sum2)

    if S_in is not None:
        S_diffout = mm_diff @ S_in
        S_sumout = mm_sum @ S_in

        return S_diffout[0] / S_sumout[0]
    else:
        return mm_diff  # / mm_sum[0, 0]


def calculate_pol_efficiency(mmQ, mmU):
    poleff_Q = np.hypot(mmQ[1], mmU[1])
    poleff_U = np.hypot(mmQ[2], mmU[2])
    poleff_QU = np.sqrt(0.5 * (mmQ[1] + mmQ[2]) ** 2 + 0.5 * (mmU[1] + mmU[2]) ** 2)
    average_poleff = np.mean((poleff_Q, poleff_U, poleff_QU))
    return average_poleff


def analyze_mm(X, X2, imr_deg):
    mmQ = model(X, X2, "0-45", imr_deg, S_in=None)[0]
    mmU = model(X, X2, "22.5-67.5", imr_deg, S_in=None)[0]
    polarimetric_efficiency = calculate_pol_efficiency(mmQ, mmU)
    return polarimetric_efficiency


def _loss_fn(X, X2, imr_angs):
    pol_effs = [analyze_mm(X, X2, imr) for imr in imr_angs]
    max_poleff = np.mean(pol_effs)
    return (1 - max_poleff) ** 2


def get_retardance_relative(filt_name: str, relative_to: float):
    lookup = {"r": 630, "i": 760, "z": 910}
    # b = lookup[relative_to]
    return 0.25 * relative_to / lookup[filt_name]


def fit_qwp_angles(input_X, filt_name: str):
    imr_angs = np.linspace(-90, -58.5, 100)
    print(f"Starting optimization for {filt_name} band")
    opt = minimize(
        _loss_fn,
        x0=(
            90,
            90,  # , 0.25
        ),
        bounds=(
            [0, 180],
            [0, 180],
            # [0.248, 0.252],
        ),
        # jac="3-point",
        args=(input_X, imr_angs),
        method="Nelder-Mead",
        options={"maxiter": 50000},
    )
    qwp1, qwp2 = opt.x
    pol_eff = 1 - np.sqrt(_loss_fn(opt.x, input_X, imr_angs))
    qwp_ret = get_retardance_relative(filt_name, qwp_design_wavelength)

    print(opt.message)
    print(
        f"{filt_name}: {qwp1=} {qwp2=} Δ={qwp_ret:.02f} (pol. eff={pol_eff * 100:.01f}%)"
    )

    return opt.x


if __name__ == "__main__":
    Xr = fit_qwp_angles(X_r, "r")
    Xi = fit_qwp_angles(X_i, "i")
    Xz = fit_qwp_angles(X_z, "z")

    imr_angs = np.linspace(-90, 0, 1000)
    measured_imr_angs = np.linspace(-90, 0, 9)

    cmap = pro.Colormap("inferno", left=0.3, right=0.7, reverse=True, discrete=True)
    colors = cmap(np.linspace(0, 1, 3))
    color_dict = {"r": colors[0], "i": colors[1], "z": colors[2]}

    fig, axes = pro.subplots(width="4in", height=f"{4 / 1.4}in")
    rows = []
    for filt_name in ("r", "i", "z"):
        

        X = eval(f"X{filt_name}")
        X2 = eval(f"X_{filt_name}")
        delta = get_retardance_relative(filt_name, qwp_design_wavelength)
        name = f"{filt_name} (θ1={X[0]:.01f}°, θ2={X[1]:.01f}°, Δ={delta:.02f}λ)"


        pol_effs = np.array(
            [analyze_mm(X, X2, imr_ang) for imr_ang in imr_angs]
        )
        axes.plot(imr_angs, pol_effs * 100, c=color_dict[filt_name], zorder=400)


        measured_pol_effs = np.array(
            [analyze_mm(X, X2, imr_ang) for imr_ang in measured_imr_angs]
        )
        axes.scatter(measured_imr_angs, measured_pol_effs * 100, c=color_dict[filt_name], label=name, zorder=500)

        mask = (imr_angs <= -58.5) & (imr_angs >= -58.5 - 30)
        ave_val = pol_effs[mask].mean()
        ave_imr = imr_angs[mask].mean()
        # axes.hlines(ave_val * 100, -90, -58.5, colors=color_dict[filt_name], lw=1, alpha=0.6)
        # axes.scatter(-90, ave_val * 100, c=color_dict[filt_name], marker="<")

        abs_peak_idx = pol_effs.argmax()
        abs_peak_imrang = imr_angs[abs_peak_idx]
        abs_peak_poleff = pol_effs[abs_peak_idx]
        # axes.axvline(abs_peak_imrang, lw=1, ls="--", c=color_dict[filt_name], zorder=300)
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
                "qwp1_deg": X[0],
                "qwp2_deg": X[1],
                "qwp_ret": delta,
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

    title = "2025/11/26 MagAO-X LP=0° with \ndouble QWP compensator and 500:1 PBS"
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

    plot_utils.save_figure(fig, paths.figures / "model_pol_eff_with_qwps_and_bs")

    df = pd.DataFrame(rows)

    df.to_csv(paths.data / "pol_eff_model_with_qwps_and_bs_report.csv", index=False)
    print(df)
