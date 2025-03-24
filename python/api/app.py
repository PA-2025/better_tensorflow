from fastapi import FastAPI, UploadFile
from typing import List
import better_tensorflow as btf
import os
from data_manager import DataManager

app = FastAPI()


@app.post("/predict_mlp")
async def predict_mlp(file: UploadFile):
    with open("temp.mp3", "wb") as f:
        f.write(await file.read())

    data = DataManager.load_data("temp.mp3")
    prediction = btf.predict_mlp(data)

    return {"prediction": prediction}


@app.post("/train_mlp")
async def training_mlp(nb_epochs: int, hidden_layers: List[int]):
    dataset_path = (
        "/home/victor/Documents/esgi/pa-2025/data-registry/script/bensound/music/"
    )
    dataset = DataManager.load_dataset(dataset_path)

    btf.train_mlp(dataset, nb_epochs, hidden_layers)

    return {"training": "OK"}
