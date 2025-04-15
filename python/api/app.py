from fastapi import FastAPI, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from typing import List
import better_tensorflow as btf
import os
import json
from data_manager import DataManager
from datetime import datetime

app = FastAPI()

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/predict_mlp")
async def predict_mlp(file: UploadFile):
    with open("temp.mp3", "wb") as f:
        f.write(await file.read())

    data = DataManager.load_data("temp.mp3")
    prediction = btf.predict_mlp(data, True)

    f = open("dataset.txt", "r")
    cat = json.loads(f.read())
    f.close()

    return {"prediction": cat[prediction]}


@app.post("/train_mlp")
async def training_mlp(nb_epochs: int, hidden_layers: List[int]):
    dataset_path = (
        "/home/victor/Documents/esgi/pa-2025/data-registry/script/scrapper/music/"
    )
    dataset = DataManager.load_dataset(dataset_path)

    now = datetime.now()

    os.makedirs("train", exist_ok=True)

    open(f"train/mlp_{now.strftime('%Y-%m-%d_%H-%M-%S')}", "a").close()

    print("Training MLP with the following parameters:")
    print(f"Number of epochs: {nb_epochs}")
    print(f"Hidden layers: {hidden_layers}")

    btf.train_mlp(
        dataset,
        [],
        nb_epochs,
        hidden_layers,
        f"train/mlp_{now.strftime('%Y-%m-%d_%H-%M-%S')}",
        True,
        True,
    )

    return {"training": "OK"}


@app.get("/get_results")
def get_results():
    files = os.listdir("train")
    files.sort(reverse=True)
    return {"files": files}


@app.get("/get_results_data")
def get_results_data():
    final = []

    files = os.listdir("train")

    for file_name in files:
        with open(f"train/{file_name}") as f:
            lines = f.readlines()

        results = []

        for line in lines:
            results.append(float(line.strip()))

        final.append({"name": file_name, "data": results})

    return {"results": final}
