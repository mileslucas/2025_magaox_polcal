import numpy as np
import pandas as pd
import paths


def doublle_diffs_and_sums(table):
    pairs = [(0, 45), (11.25, 56.25), (22.5, 67.5), (33.75, 78.75), (45, 90)]
    dfs = []
    for pair in pairs:
        table_plus = table.query(f"hwp == {pair[0]}")
        table_plus = table_plus.sort_values(["filter", "imr"])
        table_minus = table.query(f"hwp == {pair[1]}")
        table_minus = table_minus.sort_values(["filter", "imr"])

        if len(table_plus) == 0 or len(table_plus) != len(table_minus):
            continue

        double_diff = 0.5 * (
            table_plus["single_diff"].values - table_minus["single_diff"].values
        )
        double_sum = 0.5 * (
            table_plus["single_sum"].values + table_minus["single_sum"].values
        )
        double_err = 0.5 * np.hypot(
            table_plus["single_err"].values, table_minus["single_err"].values
        )

        output_table = {
            "filter": table_plus["filter"],
            "hwp": f"{pair[0]}-{pair[1]}",
            "imr": table_plus["imr"],
            "double_diff": double_diff,
            "double_sum": double_sum,
            "double_err": double_err,
            "double_norm": double_diff / double_sum,
            "double_norm_err": (
                double_diff
                / double_sum
                * np.hypot(double_err / double_diff, double_err / double_sum)
            ),
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
    dataframe = pd.read_csv(paths.data / "20251126_magaox_lp0_single_diffs.csv")
    output_df = doublle_diffs_and_sums(dataframe)
    output_df.to_csv(paths.data / "20251126_magaox_lp0_double_diffs.csv", index=False)

    dataframe = pd.read_csv(paths.data / "20251126_magaox_nolp_single_diffs.csv")
    output_df = doublle_diffs_and_sums(dataframe)
    output_df.to_csv(paths.data / "20251126_magaox_nolp_double_diffs.csv", index=False)
