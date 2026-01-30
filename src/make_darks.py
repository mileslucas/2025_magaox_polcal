from astropy.io import fits
import argparse
import pathlib
import bottleneck as bn


def make_dark(hdul):
    cube = hdul[0].data.astype("f4")
    frame = bn.nanmedian(cube, axis=0)
    return fits.PrimaryHDU(frame, header=hdul[0].header)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("filename", type=pathlib.Path)

    args = parser.parse_args()

    hdul = fits.open(args.filename)
    dark_hdu = make_dark(hdul)
    output_name = args.filename.with_stem(args.filename.stem + "_frame")
    dark_hdu.writeto(output_name, overwrite=True)
    print(f"Wrote output file to {output_name}")
