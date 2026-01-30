from astropy.io import fits
import tqdm.auto as tqdm
import argparse
import pathlib
import numpy as np
from typing import Final, Sequence
from collections import OrderedDict
import warnings
from astropy.utils.exceptions import AstropyWarning
import pandas as pd


WCS_KEYS: Final = {
    "WCSAXES",
    "CRPIX1",
    "CRPIX2",
    "CDELT1",
    "CDELT2",
    "CUNIT1",
    "CUNIT2",
    "CTYPE1",
    "CTYPE2",
    "CRVAL1",
    "CRVAL2",
    "PC1_1",
    "PC1_2",
    "PC2_1",
    "PC2_2",
}

RESERVED_KEYS: Final = {
    "NAXIS",
    "NAXIS1",
    "NAXIS2",
    "NAXIS3",
    "BSCALE",
    "BZERO",
    "BITPIX",
    "WAVEMIN",
    "WAVEMAX",
    "WAVEFWHM",
    "DLAMLAM",
} | WCS_KEYS


def dict_from_header(
    header: fits.Header, excluded=("COMMENT", "HISTORY"), fix=True
) -> OrderedDict:
    """Parse a FITS header and extract the keys and values as an ordered dictionary. Multi-line keys like ``COMMENTS`` and ``HISTORY`` will be combined with commas. The resolved path will be inserted with the ``path`` key.

    Parameters
    ----------
    header : Header
        FITS header to parse

    Returns
    -------
    OrderedDict
    """
    summary = OrderedDict()
    for k, v in header.items():
        if (
            k == ""
            or k in excluded
            or k.startswith("HOGAINCTRL")
            or k.startswith("LOGAINCTRL")
        ):
            continue
        summary[k] = v
    return summary


def combine_frames_headers(headers: Sequence[fits.Header], wcs=False):
    output_header = fits.Header()
    # let's make this easier with tables
    test_header = headers[0]
    table = pd.DataFrame([dict_from_header(header, fix=False) for header in headers])
    # use a single header to get comments
    # which columns have only a single unique value?
    unique_values = table.apply(lambda col: col.unique())
    unique_mask = unique_values.apply(lambda values: len(values) == 1)
    unique_row = table.loc[0, unique_mask]

    for key, val in unique_row.items():
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", AstropyWarning)
            output_header[key] = val, test_header.comments[key]

    # as a start, for everything else just median it
    for key in table.columns[~unique_mask]:
        if key in RESERVED_KEYS or table[key].dtype not in (int, float):
            continue
        try:
            # there is no way to check if comment exists a priori...
            comment = test_header.comments[key]
            is_err = "error" in comment
        except KeyError:
            comment = None
            is_err = False
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", AstropyWarning)
            if is_err:
                stderr = np.sqrt(np.nanmean(table[key] ** 2) / len(table))
                output_header[key] = stderr * np.sqrt(np.pi / 2), comment
            else:
                output_header[key] = np.nanmean(table[key]), comment

    ## everything below here has special rules for combinations
    # sum exposure times

    # average position using average angle formula

    wcskeys = filter(
        lambda k: any(wcsk.startswith(k) for wcsk in WCS_KEYS), output_header.keys()
    )
    for k in wcskeys:
        del output_header[k]

    return output_header


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("filenames", type=pathlib.Path, nargs="+")
    parser.add_argument("-o", "--output", type=pathlib.Path)

    args = parser.parse_args()

    pbar = tqdm.tqdm(list(sorted(args.filenames)), desc="Parsing headers")
    data = []
    headers = []
    for filename in pbar:
        frame, header = fits.getdata(filename, header=True)
        data.append(frame)
        headers.append(header)

    cube = np.array(data)
    output_header = combine_frames_headers(headers)

    fits.writeto(args.output, cube, header=output_header, overwrite=True)
    print(f"Saved file to {args.output}")
