import pathlib

path = pathlib.Path(__file__).resolve().parent.parent
dataset_path = path / "dataset"
file_path = dataset_path / "household_power_consumption .csv"
saved_model = path / "model"
model_location = saved_model / "dbscan_model.pkl"