import pandas as pd
import pocomc as pc
import numpy as np
import mueller_matrices as mm
import paths
from scipy import stats
from scipy.optimize import minimize

import corner
import matplotlib.pyplot as plt

import fit_utils

fit_A = fit_utils.BaseModel.fromdict(
    {
        "hwp_offset": [(-3, 3), 0, stats.norm(0, 6 / np.sqrt(12))],
        "hwp_retardance": [(0.46, 0.54), 0.5, stats.norm(0.5, 0.04 / np.sqrt(12))],
        "m4_diatt": [(-1, 1), 1e-2, stats.uniform(-1, 2)],
        "m4_retardance": [(0, 1), 0.5, stats.uniform(0, 1)],
        "imr_offset": [(-5, 100), 0, stats.uniform(-180, 360)],
        "imr_diatt": [(-1, 1), 1e-2, stats.uniform(-1, 2)],
        "imr_retardance_h": [(0, 1), 0.5, stats.uniform(0, 1)],
        "imr_retardance_45": [(0, 1), 0.5, stats.uniform(0, 1)],
        "imr_retardance_r": [(0, 1), 0.5, stats.uniform(0, 1)],
        "inst_offset": [(-180, 180), 0, stats.uniform(0, 360)],
        "inst_diatt": [(-1, 1), 1e-2, stats.uniform(-1, 2)],
        "inst_retardance_h": [(0, 1), 0.5, stats.uniform(0, 1)],
        "inst_retardance_45": [(0, 1), 0.5, stats.uniform(0, 1)],
        "inst_retardance_r": [(0, 1), 0.5, stats.uniform(0, 1)],
        "pbs_H_extinction": [(0, 1), 1 - 1 / 500, stats.uniform(0, 1)],
        "pbs_V_extinction": [(0, 1), 1 - 1 / 500, stats.uniform(0, 1)],
    }
)
def model(X, hwp_deg, imr_deg, S_in):
    """
    Model A is

    """
    (
        hwp_offset,
        hwp_retardance,
        m4_diatt,
        m4_retardance,
        imr_offset,
        imr_diatt,
        imr_retardance_h,
        imr_retardance_45,
        imr_retardance_r,
        inst_angle,
        inst_diatt,
        inst_retardance_h,
        inst_retardance_45,
        inst_retardance_r,
        pbs_diatt_H,
        pbs_diatt_V,
    ) = X

    hwp_theta = np.deg2rad(hwp_deg + hwp_offset)
    imr_theta = np.deg2rad(imr_deg + imr_offset)

    mm_hwp = mm.generic(hwp_theta, 0, hwp_retardance * 2 * np.pi)

    mm_m4 = mm.generic(
        theta=np.pi/2, # when plane of reflection is horizontal, the reference frame is 90°
        epsilon=m4_diatt, 
        delta=m4_retardance * 2 * np.pi
    )

    mm_imr_er = mm.elliptical_retarder(
        0,
        imr_retardance_h * 2 * np.pi,
        imr_retardance_45 * 2 * np.pi,
        imr_retardance_r * 2 * np.pi,
    )
    mm_imr_diat = mm.generic(epsilon=imr_diatt)
    R_imr = mm.rotator(imr_theta)
    mm_imr = R_imr.T @ mm_imr_er @ mm_imr_diat @ R_imr

    # mm_inst = mm.generic(np.deg2rad(inst_angle), inst_diatt, inst_retardance * 2 * np.pi)
    inst_theta = np.deg2rad(inst_angle)
    mm_inst_er = mm.elliptical_retarder(
        0,
        inst_retardance_h * 2 * np.pi,
        inst_retardance_45 * 2 * np.pi,
        inst_retardance_r * 2 * np.pi,
    )
    # inst_diatt = 0
    mm_inst_diat = mm.generic(epsilon=inst_diatt)
    R_inst = mm.rotator(inst_theta)
    mm_inst = R_inst.T @ mm_inst_er @ mm_inst_diat @ R_inst

    mm_pbs_V = mm.wollaston(ordinary=True, epsilon=pbs_diatt_V)
    mm_pbs_H = mm.wollaston(ordinary=False, epsilon=pbs_diatt_H)

    total_mm_V = mm_pbs_V @ mm_inst @ mm_imr @ mm_m4 @ mm_hwp
    total_mm_H = mm_pbs_H @ mm_inst @ mm_imr @ mm_m4 @ mm_hwp
    mm_diff = total_mm_V - total_mm_H
    mm_sum = total_mm_V + total_mm_H

    # print(f"{mm_diff}")
    S_diffout = mm_diff @ S_in
    S_sumout = mm_sum @ S_in

    # print(f"{S_diffout=} {S_sumout=}")

    return S_diffout[0] / S_sumout[0]


# def log_likelihood(y_hat, y, err):
#     sq_mahab = -0.5 * ((y - y_hat) / err) ** 2
#     prefactor = -0.5 * np.log(2 * np.pi * err**2)
#     return np.sum(prefactor * sq_mahab)


def log_likelihood(y_hat, y, err):
    sq_mahab = -0.5 * (y - y_hat) ** 2
    return np.sum(sq_mahab)


def fit_model(table_lp0, table_nolp, prior, name: str):
    hwp_angs_lp0 = table_lp0["hwp"].values
    imr_angs_lp0 = table_lp0["imr"].values
    y_values_lp0 = table_lp0["single_norm"].values
    y_err_lp0 = table_lp0["single_norm_err"].values
    y_err_lp0[y_err_lp0 == 0] = np.inf

    hwp_angs_nolp = table_nolp["hwp"].values
    imr_angs_nolp = table_nolp["imr"].values
    y_values_nolp = table_nolp["single_norm"].values
    y_err_nolp = table_nolp["single_norm_err"].values
    mask = np.isnan(y_values_nolp) | np.isnan(y_err_nolp) | (y_err_nolp == 0)
    y_values_nolp[mask] = 0
    y_err_nolp[mask] = np.inf

    def loglike(X):
        input_S = np.array([1, 1, 0, 0])
        y_hat = np.array(
            [model(X, h, i, input_S) for h, i in zip(hwp_angs_lp0, imr_angs_lp0)]
        )
        ll_lp0 = log_likelihood(y_hat, y_values_lp0, y_err_lp0)

        input_S = np.array([1, 0, 0, 0])
        y_hat = np.array(
            [model(X, h, i, input_S) for h, i in zip(hwp_angs_nolp, imr_angs_nolp)]
        )
        ll_nolp = log_likelihood(y_hat, y_values_nolp, y_err_nolp)
        # print(f"{ll_lp0=} {ll_nolp=}")
        return ll_lp0 + ll_nolp

    sampler = pc.Sampler(prior=prior, likelihood=loglike, random_state=123456)

    sampler.run()

    logz, logz_err = sampler.evidence()
    print(f"Model A | logZ: {logz} ± {logz_err}")

    samples, logl, logp = sampler.posterior(resample=True)
    np.savez_compressed(
        paths.data / f"model_A_chains_{name}.npz",
        samples=samples,
        logl=logl,
        logp=logp,
    )

    fig = corner.corner(
        samples,
        labels=[
            "HWP_off",
            "HWP_ret",
            "IMR_off",
            "IMR_diat",
            "IMR_ret_h",
            "IMR_ret_45",
            "IMR_ret_r",
            # "R_per",
            "PBS_diat_h",
        ],
        quantiles=[0.16, 0.5, 0.84],
        show_titles=True,
    )
    fig.savefig(paths.figures / f"model_A_corner_{name}.pdf")
    fig.savefig(paths.figures / f"model_A_corner_{name}.png")
    plt.show()


def fit_model_scipy(table_lp0, table_nolp, prior, name: str):
    hwp_angs_lp0 = table_lp0["hwp"].values
    imr_angs_lp0 = table_lp0["imr"].values
    y_values_lp0 = table_lp0["single_norm"].values
    y_err_lp0 = table_lp0["single_norm_err"].values
    y_err_lp0[y_err_lp0 == 0] = np.inf

    hwp_angs_nolp = table_nolp["hwp"].values
    imr_angs_nolp = table_nolp["imr"].values
    y_values_nolp = table_nolp["single_norm"].values
    y_err_nolp = table_nolp["single_norm_err"].values
    mask = np.isnan(y_values_nolp) | np.isnan(y_err_nolp) | (y_err_nolp == 0)
    y_values_nolp[mask] = 0
    y_err_nolp[mask] = np.inf

    def loss(X):
        input_S = np.array([1, 1, 0, 0])
        y_hat = np.array(
            [model(X, h, i, input_S) for h, i in zip(hwp_angs_lp0, imr_angs_lp0)]
        )
        ll_lp0 = log_likelihood(y_hat, y_values_lp0, y_err_lp0)

        input_S = np.array([1, 0, 0, 0])
        y_hat = np.array(
            [model(X, h, i, input_S) for h, i in zip(hwp_angs_nolp, imr_angs_nolp)]
        )
        ll_nolp = log_likelihood(y_hat, y_values_nolp, y_err_nolp)
        # print(f"{ll_lp0=} {ll_nolp=}")
        return -1 * (ll_lp0 + ll_nolp)

    if name == "r":
        X0 = [
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

    elif name == "i":
        X0 = [
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
    elif name == "z":
        X0 = [
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
        
    opt = minimize(
        loss,
        x0=fit_A.init,
        bounds=fit_A.limits,
        method="Nelder-Mead",
        # jac="3-point",
        # hess="3-point",
        options={"maxiter": 100000},
    )
    print(opt.message)
    print("Model A params:")
    for p in opt.x:
        print(f"{p},")
    print(opt.x)


if __name__ == "__main__":
    table_lp0 = pd.read_csv(paths.data / "20251126_magaox_lp0_single_diffs.csv")
    table_nolp = pd.read_csv(paths.data / "20251126_magaox_nolp_single_diffs.csv")

    r_table_lp0 = table_lp0.query("filter == 'r'")
    i_table_lp0 = table_lp0.query("filter == 'i'")
    z_table_lp0 = table_lp0.query("filter == 'z'")

    r_table_nolp = table_nolp.query("filter == 'r'")
    i_table_nolp = table_nolp.query("filter == 'i'")
    z_table_nolp = table_nolp.query("filter == 'z'")

    prior = pc.Prior(
        [
            stats.norm(0, 10),  # hwp_offset
            stats.norm(180, 10),  # hwp_retardance
            stats.uniform(-180, 360),  # imr_offset
            stats.uniform(0, 1),  # imr_diatt
            stats.uniform(-180, 360),  # imr_retardance_h
            stats.uniform(-180, 360),  # imr_retardance_45
            stats.uniform(-180, 360),  # imr_retardance_r
            stats.norm(14, 10),  # periscope angle
            stats.uniform(0.5, 0.5),  # pbs_through_o
            stats.uniform(0.5, 0.5),  # pbs_through_e
        ]
    )

    print("Optimizing r")
    fit_model_scipy(r_table_lp0, r_table_nolp, prior, name="r")
    print("Optimizing i")
    fit_model_scipy(i_table_lp0, i_table_nolp, prior, name="i")
    print("Optimizing z")
    fit_model_scipy(z_table_lp0, z_table_nolp, prior, name="z")
