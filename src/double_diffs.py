import numpy as np
import pandas as pd
import paths


def doublle_diffs_and_sums(table):
    pairs = [(0, 45), (11.25, 56.25), (22.5, 67.5), (33.75, 78.75), (45, 90)]
    dfs = []
    for pair in pairs:
        table_plus = table.query(f"hwp_ang == {pair[0]}")
        table_plus = table_plus.sort_values("imr_ang")
        table_minus = table.query(f"hwp_ang == {pair[1]}")
        table_minus = table_minus.sort_values("imr_ang")

        if len(table_plus) == 0 or len(table_plus) != len(table_minus):
            continue

        double_diff = 0.5 * (
            table_plus["single_diff"].values - table_minus["single_diff"].values
        )
        double_err = 0.5 * np.hypot(
            table_plus["single_diff_err"].values, table_minus["single_diff_err"].values
        )

        output_table = {
            "hwp_ang": f"{pair[0]}-{pair[1]}",
            "imr_ang": table_plus["imr_ang"],
            "double_diff": double_diff,
            "double_diff_err": double_err,
        }
        dfs.append(pd.DataFrame(output_table))

    return pd.concat(dfs)


# def double_diffs_and_sums(table):
#     table_hwp0 = table.query("hwp == 0")
#     table_hwp1125 = table.query("hwp == 11.75")
#     table_hwp225 = table.query("hwp == 22.5")
#     table_hwp3375 = table.query("hwp == 33.25")
#     table_hwp45 = table.query("hwp == 45.0")
#     table_hwp5625 = table.query("hwp == 56.75")
#     table_hwp675 = table.query("hwp == 67.5")
#     table_hwp7875 = table.query("hwp == 78.75")
#     table_hwp90 = table.query("hwp == 90.0")


if __name__ == "__main__":

    table_r = pd.read_csv(paths.data / "20260321_pdi_calib_lpV_r.csv")
    table_i = pd.read_csv(paths.data / "20260321_pdi_calib_lpV_i.csv")
    table_z = pd.read_csv(paths.data / "20260321_pdi_calib_lpV_z.csv")

    table_double_r = doublle_diffs_and_sums(table_r)
    table_double_i = doublle_diffs_and_sums(table_i)
    table_double_z = doublle_diffs_and_sums(table_z)

    table_double_r.to_csv(paths.data / "20260321_pdi_calib_lpV_double-diff_r.csv", index=False)
    table_double_i.to_csv(paths.data / "20260321_pdi_calib_lpV_double-diff_i.csv", index=False)
    table_double_z.to_csv(paths.data / "20260321_pdi_calib_lpV_double-diff_z.csv", index=False)

    table_r = pd.read_csv(paths.data / "20260321_pdi_calib_nolp_r.csv")
    table_i = pd.read_csv(paths.data / "20260321_pdi_calib_nolp_i.csv")
    table_z = pd.read_csv(paths.data / "20260321_pdi_calib_nolp_z.csv")

    table_double_r = doublle_diffs_and_sums(table_r)
    table_double_i = doublle_diffs_and_sums(table_i)
    table_double_z = doublle_diffs_and_sums(table_z)

    table_double_r.to_csv(paths.data / "20260321_pdi_calib_nolp_double-diff_r.csv", index=False)
    table_double_i.to_csv(paths.data / "20260321_pdi_calib_nolp_double-diff_i.csv", index=False)
    table_double_z.to_csv(paths.data / "20260321_pdi_calib_nolp_double-diff_z.csv", index=False)