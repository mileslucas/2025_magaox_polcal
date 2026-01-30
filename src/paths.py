import pathlib

root = pathlib.Path(__file__).parents[1]
src = root / "src"
data = root / "data"
figures = root / "figures"

data.mkdir(exist_ok=True)
figures.mkdir(exist_ok=True)
