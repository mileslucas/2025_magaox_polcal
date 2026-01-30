import pandas as pd
import pocomc as pc
import numpy as np
import mueller_matrices as mm
import paths
from scipy import stats
from scipy.optimize import minimize
import fit_utils

import corner
import matplotlib.pyplot as plt


fit_B = fit_utils.BaseModel.fromdict(
    {
        "hwp_offset": [(-3, 3), 0, stats.norm(0, 6 / np.sqrt(12))],
        "hwp_retardance": [(0.46, 0.54), 0.5, stats.norm(0.5, 0.04 / np.sqrt(12))],
        # "m4_diatt": [(-1, 1), 1e-2, stats.uniform(-1, 2)],
        "m4_retardance": [(0, 1), 0.5, stats.uniform(0, 1)],
        "imr_offset": [(-5, 100), 0, stats.uniform(-180, 360)],
        "imr_retardance_h": [(0, 1), 0.5, stats.uniform(0, 1)],
        "imr_retardance_45": [(0, 1), 0.5, stats.uniform(0, 1)],
        "imr_retardance_r": [(0, 1), 0.5, stats.uniform(0, 1)],
        "inst_offset": [(-180, 180), 0, stats.uniform(0, 360)],
        "inst_retardance_h": [(0, 1), 0.5, stats.uniform(0, 1)],
        "inst_retardance_45": [(0, 1), 0.5, stats.uniform(0, 1)],
        "inst_retardance_r": [(0, 1), 0.5, stats.uniform(0, 1)],
        "pbs_extinction": [(0, 1), 1 - 1 / 500, stats.uniform(0, 1)],
    }
)


def model(X, hwp_str, imr_deg, S_in):
    """
    Model B is

    """
    (
        hwp_offset,
        hwp_retardance,
        m4_retardance,
        imr_offset,
        imr_retardance_h,
        imr_retardance_45,
        imr_retardance_r,
        inst_offset,
        inst_retardance_h,
        inst_retardance_45,
        inst_retardance_r,
        pbs_extinction,
    ) = X

    hwp_str1, hwp_str2 = hwp_str.split("-")
    hwp_deg1 = float(hwp_str1)
    hwp_deg2 = float(hwp_str2)
    hwp_theta1 = np.deg2rad(hwp_deg1 + hwp_offset)
    hwp_theta2 = np.deg2rad(hwp_deg2 + hwp_offset)
    imr_theta = np.deg2rad(imr_deg + imr_offset)

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
    m4_diatt = 0
    mm_m4 = mm.generic(
        theta=np.pi/2, # when plane of reflection is horizontal, the reference frame is 90°
        epsilon=m4_diatt, 
        delta=m4_retardance * 2 * np.pi
    )

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
    inst_theta = np.deg2rad(inst_offset)
    mm_inst = mm.elliptical_retarder(
        inst_theta,
        inst_retardance_h * 2 * np.pi,
        inst_retardance_45 * 2 * np.pi,
        inst_retardance_r * 2 * np.pi,
    )
    # mm_inst = mm.generic(
    #     inst_theta,
    #     0,
    #     inst_retardance_h * 2 * np.pi
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

    total_mm_V1 = mm_pbs_V @ mm_inst @ mm_imr @ mm_m4 @ mm_hwp1
    total_mm_H1 = mm_pbs_H @ mm_inst @ mm_imr @ mm_m4 @ mm_hwp1
    mm_diff1 = total_mm_V1 - total_mm_H1
    mm_sum1 = total_mm_V1 + total_mm_H1

    total_mm_V2 = mm_pbs_V @ mm_inst @ mm_imr @ mm_m4 @ mm_hwp2
    total_mm_H2 = mm_pbs_H @ mm_inst @ mm_imr @ mm_m4 @ mm_hwp2
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


# def log_likelihood(y_hat, y, err):
#     sq_mahab = -0.5 * ((y - y_hat) / err) ** 2
#     prefactor = -0.5 * np.log(2 * np.pi * err**2)
#     return np.sum(prefactor * sq_mahab)


def log_likelihood(y_hat, y, err):
    sq_mahab = -0.5 * (y - y_hat) ** 2
    return np.sum(sq_mahab)


def fit_model(table_lp0, name: str):
    hwp_angs_lp0 = table_lp0["hwp"].values
    imr_angs_lp0 = table_lp0["imr"].values
    y_values_lp0 = table_lp0["double_norm"].values
    y_err_lp0 = table_lp0["double_norm_err"].values
    y_err_lp0[y_err_lp0 == 0] = np.inf

    def loglike(X):
        input_S = np.array([1, 1, 0, 0])
        if X[-2] + X[-1] > 1:
            return -np.inf
        y_hat = np.array(
            [model(X, h, i, input_S) for h, i in zip(hwp_angs_lp0, imr_angs_lp0)]
        )
        ll_lp0 = log_likelihood(y_hat, y_values_lp0, y_err_lp0)
        return ll_lp0

    sampler = pc.Sampler(prior=fit_B.prior, likelihood=loglike, random_state=123456)

    sampler.run()

    logz, logz_err = sampler.evidence()
    print(f"Model B | logZ: {logz} ± {logz_err}")

    samples, logl, logp = sampler.posterior(resample=True)
    np.savez_compressed(
        paths.data / f"model_B_chains_{name}.npz",
        samples=samples,
        logl=logl,
        logp=logp,
    )

    fig = corner.corner(
        samples,
        labels=fit_B.names,
        quantiles=[0.16, 0.5, 0.84],
        show_titles=True,
    )
    fig.savefig(paths.figures / f"model_B_corner_{name}.pdf")
    fig.savefig(paths.figures / f"model_B_corner_{name}.png")
    plt.show()


def fit_model_scipy(table_lp0, name: str):
    hwp_angs_lp0 = table_lp0["hwp"].values
    imr_angs_lp0 = table_lp0["imr"].values
    y_values_lp0 = table_lp0["double_norm"].values
    y_err_lp0 = table_lp0["double_norm_err"].values
    y_err_lp0[y_err_lp0 == 0] = np.inf

    def loss(X):
        input_S = np.array([1, 1, 0, 0])
        y_hat = np.array(
            [model(X, h, i, input_S) for h, i in zip(hwp_angs_lp0, imr_angs_lp0)]
        )
        ll_lp0 = log_likelihood(y_hat, y_values_lp0, y_err_lp0)
        return -ll_lp0

    opt = minimize(
        loss,
        x0=fit_B.init,
        bounds=fit_B.limits,
        method="Nelder-Mead",
        jac="3-point",
        hess="3-point",
        options={"maxiter": 100000},
    )

    print(f"Model A params for filter {name}:")
    for p in opt.x:
        print(f"{p},")
    print()
    print(opt.x)


if __name__ == "__main__":
    table_lp0 = pd.read_csv(paths.data / "20251126_magaox_lp0_double_diffs.csv")
    table_nolp = pd.read_csv(paths.data / "20251126_magaox_nolp_double_diffs.csv")

    r_table_lp0 = table_lp0.query("filter == 'r'")
    i_table_lp0 = table_lp0.query("filter == 'i'")
    z_table_lp0 = table_lp0.query("filter == 'z'")

    r_table_nolp = table_nolp.query("filter == 'r'")
    i_table_nolp = table_nolp.query("filter == 'i'")
    z_table_nolp = table_nolp.query("filter == 'z'")

    # fit_model(i_table_lp0, name="i")
    fit_model_scipy(r_table_lp0, name="r")
    fit_model_scipy(i_table_lp0, name="i")
    fit_model_scipy(z_table_lp0, name="z")
