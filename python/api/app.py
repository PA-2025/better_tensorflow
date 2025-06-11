from fastapi import FastAPI, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from typing import List
import better_tensorflow as btf
import json
from data_manager import DataManager
from datetime import datetime
from database_manager import DatabaseManager

app = FastAPI()

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DATASET_PATH = "python/api/data/music_spec/"


@app.post("/predict_rbf")
async def predict_rbf(file: UploadFile):
    with open("temp.mp3", "wb") as f:
        f.write(await file.read())

    data = DataManager.load_data("temp.mp3")
    data = btf.convert_image_to_array(data)
    prediction = btf.predict_rbf(data, True, True)

    f = open("dataset.txt", "r")
    cat = json.loads(f.read())
    f.close()

    return {"prediction": cat[prediction]}


@app.post("/train_rbf")
async def training_rbf(
    nb_epochs: int,
    hidden_layers: List[int],
    learning_rate: float,
    filter_cat: List[str],
    nb_epoch_to_save: int = 10000,
):
    dataset, dataset_test = DataManager.load_dataset(DATASET_PATH, filter_cat)

    now = datetime.now()

    btf.train_rbf(
        dataset,
        dataset_test,
        nb_epochs,
        hidden_layers[0],
        f"train/mlp_{now.strftime('%Y-%m-%d_%H-%M-%S')}",
        True,
        True,
        True,
        learning_rate=learning_rate,
        nb_epoch_to_save=nb_epoch_to_save,
    )

    return {"training": "OK"}


@app.post("/predict_mlp")
async def predict_mlp(file: UploadFile):
    with open("temp.mp3", "wb") as f:
        f.write(await file.read())

    data = DataManager.load_data("temp.mp3")
    data = btf.convert_image_to_array(data)
    prediction = btf.predict_mlp(data, [], True, True)

    f = open("dataset.txt", "r")
    cat = json.loads(f.read())
    f.close()

    return {"prediction": cat[prediction]}


@app.post("/train_mlp")
async def training_mlp(
    nb_epochs: int,
    hidden_layers: List[int],
    learning_rate: float,
    filter_cat: List[str],
    nb_epoch_to_save: int = 10000,
):
    dataset, dataset_test = DataManager.load_dataset(DATASET_PATH, filter_cat)

    now = datetime.now()

    btf.train_mlp(
        dataset,
        dataset_test,
        [],
        nb_epochs,
        hidden_layers,
        f"train/mlp_{now.strftime('%Y-%m-%d_%H-%M-%S')}",
        True,
        True,
        True,
        learning_rate=learning_rate,
        nb_epoch_to_save=nb_epoch_to_save,
    )

    return {"training": "OK"}


@app.get("/get_results")
def get_results():
    return {
        "files": DatabaseManager.get_results() + DatabaseManager.get_results_mongo()
    }


@app.get("/get_results_data")
def get_results_data():
    return {
        "results": DatabaseManager.get_training_data()
        + DatabaseManager.get_training_data_mongo()
    }


@app.get("/get_dataset_cat")
def get_dataset_cat():
    return {"cat": DataManager.find_dataset_categories(DATASET_PATH)}
