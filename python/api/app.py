import os

import numpy as np
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


@app.post("/predict_all")
async def predict_all(file: UploadFile, weight_file: UploadFile):
    with open("temp.mp3", "wb") as f:
        f.write(await file.read())

    with open("weight_temp", "wb") as f:
        f.write(await weight_file.read())

    data = DataManager.load_data("temp.mp3")

    algo = DataManager.find_weight_category("weight_temp")
    DataManager.convert_file_good_weight("weight_temp", algo)

    results = []
    for d in data:
        prediction = 0
        match algo:
            case "rbf":
                array = btf.convert_matrix_to_array(d.tolist())
                prediction = btf.predict_rbf(array, True, True)
            case "mlp":
                array = btf.convert_matrix_to_array(d.tolist())
                prediction = btf.predict_mlp(array, [], True, True)
            case "svm":
                array = btf.convert_matrix_to_array(d.tolist())
                svm=btf.kernelSVM("rbf", 2.0, lr=0.1, lambda_svm=0.01, epochs=200)
                files=os.listdir("svm_*.weight")
                for file in files:
                    os.rename(file,"wsvm.weight")
                    prediction = svm.predict(array)
                    if prediction == 1:
                        break

        results.append(prediction)

    f = open("dataset.txt", "r")
    cat = json.loads(f.read())
    f.close()

    return {"prediction": cat[np.argmax(results)]}


@app.post("/train_rbf")
async def training_rbf(
    gamma: float,
    number_clusters: int,
    filter_cat: List[str],
):
    dataset, dataset_test = DataManager.load_dataset(DATASET_PATH, filter_cat)

    now = datetime.now()

    btf.train_rbf(
        dataset,
        dataset_test,
        [],
        number_clusters,
        gamma,
        True,
        True,
        f"train/mlp_{now.strftime('%Y-%m-%d_%H-%M-%S')}",
    )

    return {"training": "OK"}


@app.post("/predict_mlp")
async def predict_mlp(file: UploadFile):
    with open("temp.mp3", "wb") as f:
        f.write(await file.read())

    data = DataManager.load_data("temp.mp3")
    results = []
    for d in data:
        array = btf.convert_matrix_to_array(d.tolist())
        prediction = btf.predict_mlp(array, [], True, True)
        results.append(prediction)

    f = open("dataset.txt", "r")
    cat = json.loads(f.read())
    f.close()

    print(results)

    return {"prediction": cat[np.argmax(results)]}


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
        False,
        True,
        learning_rate=learning_rate,
        nb_epoch_to_save=nb_epoch_to_save,
    )

    return {"training": "OK"}

@app.post("/train_svm")
async def training_svm(
        nb_epochs: int,
        learning_rate: float,
        filter_cat: List[str],
        lambda_svm : float,
        kernel: str,
        bias: float ,
        nb_epoch_to_save: int = 10000,

):
    dataset, dataset_test = DataManager.load_dataset(DATASET_PATH, filter_cat)
    datasets_y = []
    for k in range(len(dataset)):
        dataset_y = []
        for i in range(len(dataset)):
                for j in range(len(dataset[i])):
                    if i == k:
                        dataset_y.append(0)
                    else:
                        dataset_y.append(1)
        datasets_y.append(dataset_y)

    now = datetime.now()

    os.remove('svm_*.weight')

    svm = btf.KernelSVM(kernel, bias, lr=learning_rate, lambda_svm=lambda_svm, epochs=nb_epochs)
    for i in range(len(datasets_y) - 1 if len(datasets_y)<=2 else 0):
        svm.fit(dataset,datasets_y[i],f"svm_{i}")

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
