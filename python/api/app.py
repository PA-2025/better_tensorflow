from fastapi import FastAPI, UploadFile
from typing import List
import better_tensorflow as btf
from data_manager import DataManager

app = FastAPI()


@app.post("/predict_perceptron")
async def create_file(file: UploadFile):
    with open("temp.mp3", "wb") as f:
        f.write(await file.read())

    data = DataManager.load_data("temp.mp3")
    prediction = btf.predict_perceptron(data)

    return {"prediction": prediction}
