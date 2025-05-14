from fastapi import FastAPI, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from typing import List
import better_tensorflow as btf
import os
import json
from data_manager import DataManager
from datetime import datetime
import sqlite3

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
):
    print(hidden_layers)

    dataset = DataManager.load_dataset(DATASET_PATH, filter_cat)
    dataset_test = DataManager.load_dataset(DATASET_PATH, filter_cat)

    now = datetime.now()

    print("Training MLP with the following parameters:")
    print(f"Number of epochs: {nb_epochs}")
    print(f"Hidden layers: {hidden_layers}")

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
    )

    return {"training": "OK"}


@app.get("/get_results")
def get_results():
    con = sqlite3.connect("app_database.db")
    cur = con.cursor()
    res = cur.execute("select distinct training_name from training_data ")
    rows = res.fetchall()
    results = [row[0] for row in rows]
    files = []
    for r in results:
        files.append(f"{r}_mse")
        files.append(f"{r}_accuracy")
    return {"files": files}


@app.get("/get_results_data")
def get_results_data():
    con = sqlite3.connect("app_database.db")
    cur = con.cursor()
    res = cur.execute("select distinct training_name from training_data ")
    rows = res.fetchall()
    names = [row[0] for row in rows]
    final = []

    for file_name in names:
        results = []
        res = cur.execute(
            "select mse, accuracy from training_data  where training_name = ? order by epoch",
            (file_name,),
        )
        for line in res.fetchall():
            results.append(
                {
                    "mse": line[0],
                    "accuracy": line[1],
                }
            )

        final.append({"name": f"{file_name}_mse", "data": [x["mse"] for x in results]})
        final.append(
            {"name": f"{file_name}_accuracy", "data": [x["accuracy"] for x in results]}
        )

    return {"results": final}


@app.get("/get_dataset_cat")
def get_dataset_cat():
    return {"cat": DataManager.find_dataset_categories(DATASET_PATH)}
