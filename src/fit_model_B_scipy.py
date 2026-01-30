import pandas as pd
import pocomc as pc
import numpy as np
import mueller_matrices as mm
import paths
from scipy import stats

import corner
import matplotlib.pyplot as plt


def model(X, hwp_str, imr_deg, S_in):
    """
    Model B is

    """
    (
        hwp_offset,
        hwp_retardance,
        imr_offset,
        imr_retardance_h,
        imr_retardance_45,
        imr_retardance_r,
        periscope_angle,
        # inst_diatt,
        pbs_diatt_H,
        pbs_diatt_V,
    ) = X

    hwp_str1, hwp_str2 = hwp_str.split("-")
    hwp_deg1 = float(hwp_str1)
    hwp_deg2 = float(hwp_str2)
    hwp_theta1 = np.deg2rad(hwp_deg1 + hwp_offset)
    hwp_theta2 = np.deg2rad(hwp_deg2 + hwp_offset)
    imr_theta = np.deg2rad(imr_deg + imr_offset)

    mm_hwp1 = mm.generic(hwp_theta1, 0, np.deg2rad(hwp_retardance))
    mm_hwp2 = mm.generic(hwp_theta2, 0, np.deg2rad(hwp_retardance))

    mm_imr = mm.elliptical_retarder(
        imr_theta,
        np.deg2rad(imr_retardance_h),
        np.deg2rad(imr_retardance_45),
        np.deg2rad(imr_retardance_r),
    )
    T_periscope = mm.rotator(np.deg2rad(periscope_angle))
    mm_pbs_V = mm.generic(theta=0, epsilon=pbs_diatt_V)
    mm_pbs_H = mm.generic(theta=np.pi / 2, epsilon=pbs_diatt_H)

    total_mm_V1 = mm_pbs_V @ T_periscope @ mm_imr @ mm_hwp1
    total_mm_H1 = mm_pbs_H @ T_periscope @ mm_imr @ mm_hwp1
    mm_diff1 = total_mm_V1 - total_mm_H1
    mm_sum1 = total_mm_V1 + total_mm_H1

    total_mm_V2 = mm_pbs_V @ T_periscope @ mm_imr @ mm_hwp2
    total_mm_H2 = mm_pbs_H @ T_periscope @ mm_imr @ mm_hwp2
    mm_diff2 = total_mm_V2 - total_mm_H2
    mm_sum2 = total_mm_V2 + total_mm_H2

    mm_diff = 0.5 * (mm_diff1 - mm_diff2)
    mm_sum = 0.5 * (mm_sum1 + mm_sum2)

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
    y_values_lp0 = table_lp0["double_norm"].values
    y_err_lp0 = table_lp0["double_norm_err"].values
    y_err_lp0[y_err_lp0 == 0] = np.inf

    hwp_angs_nolp = table_nolp["hwp"].values
    imr_angs_nolp = table_nolp["imr"].values
    y_values_nolp = table_nolp["double_norm"].values
    y_err_nolp = table_nolp["double_norm_err"].values
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
        labels=[
            "HWP_off",
            "HWP_ret",
            "IMR_off",
            "IMR_ret_h",
            "IMR_ret_45",
            "IMR_ret_r",
            "R_per",
            "PBS_diat_h",
            "PBS_diat_v",
        ],
        quantiles=[0.16, 0.5, 0.84],
        show_titles=True,
    )
    fig.savefig(paths.figures / f"model_B_corner_{name}.pdf")
    fig.savefig(paths.figures / f"model_B_corner_{name}.png")
    plt.show()


if __name__ == "__main__":
    table_lp0 = pd.read_csv(paths.data / "20251126_magaox_lp0_double_diffs.csv")
    table_nolp = pd.read_csv(paths.data / "20251126_magaox_nolp_double_diffs.csv")

    r_table_lp0 = table_lp0.query("filter == 'r'")
    i_table_lp0 = table_lp0.query("filter == 'i'")
    z_table_lp0 = table_lp0.query("filter == 'z'")

    r_table_nolp = table_nolp.query("filter == 'r'")
    i_table_nolp = table_nolp.query("filter == 'i'")
    z_table_nolp = table_nolp.query("filter == 'z'")

    prior = pc.Prior(
        [
            stats.norm(0, 10),  # hwp_offset
            stats.norm(180, 40),  # hwp_retardance
            stats.uniform(-180, 360),  # imr_offset
            stats.uniform(-180, 360),  # imr_retardance_h
            stats.uniform(-180, 360),  # imr_retardance_45
            stats.uniform(-180, 360),  # imr_retardance_r
            stats.uniform(-20, 40),  # periscope angle
            stats.uniform(0.5, 0.5),  # pbs_through_o
            stats.uniform(0.5, 0.5),  # pbs_through_e
        ]
    )

    fit_model(i_table_lp0, i_table_nolp, prior, name="i")
