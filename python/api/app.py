import os
import numpy as np
import shutil
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
                scores = []
                files = sorted(
                    [
                        f
                        for f in os.listdir()
                        if f.startswith("svm_") and f.endswith(".weights")
                    ]
                )
                for file in files:
                    svm = btf.KernelSVM("rbf", 2.0, lr=0.1, lambda_svm=0.01, epochs=200)
                    svm.load_weights_from(file)
                    pred = svm.predict([array])[0]
                    scores.append(pred)
                prediction = 0
                for i in range(len(scores)):
                    if scores[i] == 1:
                        prediction = i
                        break

        results.append(prediction)

    with open("dataset.txt", "r") as f:
        cat = json.load(f)

    return {"prediction": cat[int(max(set(results), key=results.count))]}


@app.post("/train_rbf")
async def training_rbf(
    gamma: float,
    number_clusters: int,
    filter_cat: List[str],
):
    dataset, dataset_test = DataManager.load_dataset(DATASET_PATH, filter_cat)

    now = datetime.now()

    r = btf.train_rbf(
        dataset,
        dataset_test,
        [],
        number_clusters,
        gamma,
        True,
        True,
        f"train/mlp_{now.strftime('%Y-%m-%d_%H-%M-%S')}",
    )
    print(r)

    return {"training": "OK"}


@app.post("/predict_mlp")
async def predict_mlp(file: UploadFile):
    with open("temp.mp3", "wb") as f:
        f.write(await file.read())

    data = DataManager.load_data("temp.mp3")
    results = []
    weights = [
        "classical_hip-hop_jazz_mlp.weight",
        "pop_rock_mlp.weight",
        "techno_wajnberg_mlp.weight",
    ]
    for d in data:
        array = btf.convert_matrix_to_array(d.tolist())
        for i in range(len(weights)):
            shutil.copy(weights[i], "w_mlp.weight")
            prediction = btf.predict_mlp(array, [], True, False)
            prediction = int(prediction) if i == 0 else int(prediction) + i * 2 + 1
            results.append(prediction)

    with open("dataset.txt", "r") as f:
        cat = json.load(f)

    print(results)

    return {"prediction": cat[int(max(set(results), key=results.count))]}


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
    param: float,
    learning_rate: float,
    filter_cat: List[str],
    lambda_svm: float,
    kernel: str,
):
    import numpy as np

    dataset, dataset_test = DataManager.load_dataset(DATASET_PATH, filter_cat)

    # Création des labels One-vs-All
    datasets_y = []
    for k in range(len(dataset)):
        dataset_y = []
        for i in range(len(dataset)):
            for j in range(len(dataset[i])):
                if i == k:
                    dataset_y.append(1)
                else:
                    dataset_y.append(-1)
        datasets_y.append(dataset_y)

    # Supprime les anciens fichiers de poids SVM
    for file in os.listdir():
        if file.startswith("svm_") and file.endswith(".weights"):
            os.remove(file)

    # Prépare les données de validation
    x_val = [item for sublist in dataset_test for item in sublist]
    y_val = [i for i, sublist in enumerate(dataset_test) for _ in sublist]

    # Entraînement One-vs-All
    svm = btf.KernelSVM(
        kernel, param, lr=learning_rate, lambda_svm=lambda_svm, epochs=nb_epochs
    )
    for i in range(len(datasets_y)):
        x_data = np.array(
            [item for sublist in dataset for item in sublist], dtype=np.float64
        )
        y_data = np.array(datasets_y[i], dtype=np.float64)

        # Appliquer +1 / -1 aussi sur les labels de validation
        y_val_bin = [1 if label == i else -1 for label in y_val]

        svm.fit(
            x_data.tolist(),
            y_data.astype(int).tolist(),
            f"svm_{i}.weights",
            x_val,
            y_val_bin,  # Labels de validation pour la validation croisée
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
