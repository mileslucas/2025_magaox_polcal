import numpy as np
import pandas as pd
import paths


def single_diffs_and_sums(table):
    table_cam1 = table.query("camera == 'camsci1'")
    table_cam1 = table_cam1.sort_values(["filter", "hwp", "imr"])
    table_cam2 = table.query("camera == 'camsci2'")
    table_cam2 = table_cam2.sort_values(["filter", "hwp", "imr"])

    single_diff = table_cam2["mean"].values - table_cam1["mean"].values
    single_sum = table_cam2["mean"].values + table_cam1["mean"].values
    single_err = np.hypot(table_cam2["stderr"].values, table_cam1["stderr"].values)

    output_table = table_cam1.drop("camera", axis="columns")
    output_table = {
        "filter": table_cam1["filter"],
        "hwp": table_cam1["hwp"],
        "imr": table_cam1["imr"],
        "single_diff": single_diff,
        "single_sum": single_sum,
        "single_err": single_err,
        "single_norm": single_diff / single_sum,
        "single_norm_err": (
            single_diff
            / single_sum
            * np.hypot(single_err / single_diff, single_err / single_sum)
        ),
    }

    return pd.DataFrame(output_table)


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
    dataframe = pd.read_csv(paths.data / "20251126_magaox_lp0.csv")
    output_df = single_diffs_and_sums(dataframe)
    output_df.to_csv(paths.data / "20251126_magaox_lp0_single_diffs.csv", index=False)

    dataframe = pd.read_csv(paths.data / "20251126_magaox_nolp.csv")
    output_df = single_diffs_and_sums(dataframe)
    output_df.to_csv(paths.data / "20251126_magaox_nolp_single_diffs.csv", index=False)
