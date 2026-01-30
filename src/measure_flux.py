from astropy.io import fits
import tqdm.auto as tqdm
import argparse
import pathlib
import numpy as np
import pandas as pd


def crop_frame(frame):
    # take top left 600x600 to avoid vignetting
    return frame[-600:, :600]


def measure_flux_and_err(hdul):
    frame = crop_frame(hdul[0].data)
    frame_err = crop_frame(hdul[1].data)

    weights = 1 / frame_err**2
    mean = np.sum(weights * frame) / np.sum(weights)
    stderr = np.sqrt(1 / np.sum(weights))

    return mean, stderr


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("filenames", type=pathlib.Path, nargs="+")
    parser.add_argument("-o", "--output", type=pathlib.Path, default="table.csv")

    args = parser.parse_args()

    pbar = tqdm.tqdm(args.filenames)

    rows = []
    for filename in pbar:
        hdul = fits.open(filename)
        header = hdul[0].header
        mean, se = measure_flux_and_err(hdul)
        cam_num = header["CAMERA"][-1]
        rows.append(
            {
                "filter": header[f"hierarch FWSCI{cam_num} PRESET NAME"],
                "camera": header["CAMERA"],
                "hwp": np.round(header["hierarch HWPTRACK ACTUAL ANGLE"], 2),
                "imr": np.round(header["hierarch STAGEK POS"], 2),
                "mean": mean,
                "stderr": se,
            }
        )

    dataframe = pd.DataFrame(rows)
    dataframe.sort_values(["filter", "camera", "hwp", "imr"], inplace=True)
    dataframe.to_csv(args.output, index=False)
