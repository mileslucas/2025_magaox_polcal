from astropy.io import fits
import tqdm.auto as tqdm
import argparse
import pathlib


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("filenames", type=pathlib.Path, nargs="+")
    parser.add_argument("-o", "--output", type=pathlib.Path, default=pathlib.Path.cwd())
    parser.add_argument("-c/-nc", "--copy/--no-copy", default=True)

    args = parser.parse_args()

    pbar = tqdm.tqdm(args.filenames, desc="Parsing headers")
    for filename in pbar:
        header = fits.getheader(filename)
        output_folder = determine_folder(header)
        if args.copy:
            ...
        else:
            ...
