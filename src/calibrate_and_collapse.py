from astropy.io import fits
import tqdm.auto as tqdm
import argparse
import pathlib
import bottleneck as bn
import numpy as np


def get_key_from_header(header):
    camera = header["CAMERA"]
    exptime = header[f"hierarch {camera.upper()} EXPTIME"]
    emgain = header[f"hierarch {camera.upper()} EMGAIN"]
    return camera, exptime, emgain


def prepare_darks(dark_hduls):
    output_dict = {}
    for dark_hdul in dark_hduls:
        header = dark_hdul[0].header
        key = get_key_from_header(header)
        output_dict[key] = dark_hdul

    return output_dict


def median_and_stderr(cube):
    # Median along the stack axis
    median_img = bn.median(cube, axis=0)

    # Number of images
    N = cube.shape[0]

    sigma = np.std(cube, axis=0, ddof=1)
    sem_median = sigma * np.sqrt(np.pi / (2 * N))

    return median_img, sem_median


def calibrate(hdul, darks_dict):
    header = hdul[0].header
    key = get_key_from_header(header)
    matching_dark_hdul = darks_dict[key]

    cube = hdul[0].data.astype("f4")
    dark_subbed = cube - matching_dark_hdul[0].data
    frame, frame_err = median_and_stderr(dark_subbed)

    hdul = fits.HDUList(
        [
            fits.PrimaryHDU(frame, header=header),
            fits.ImageHDU(frame_err, header=header, name="ERR"),
        ]
    )
    return hdul


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("filenames", type=pathlib.Path, nargs="+")
    parser.add_argument("-o", "--outdir", type=pathlib.Path, default=pathlib.Path.cwd())

    parser.add_argument("-d", "--darks", type=pathlib.Path)

    args = parser.parse_args()

    pbar = tqdm.tqdm(args.filenames)

    dark_hduls = [fits.open(f) for f in args.darks.glob("*.fits")]
    darks_dict = prepare_darks(dark_hduls)

    for filename in pbar:
        hdul = fits.open(filename)
        output_hdul = calibrate(hdul, darks_dict)
        output_name = filename.name.replace(".fits", "_calib_frame.fits")
        args.outdir.mkdir(parents=True, exist_ok=True)
        output_hdul.writeto(args.outdir / output_name, overwrite=True)
        pbar.write(f"Wrote output file to {output_name}")
