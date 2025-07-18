import os
import random
import shutil
from fastapi import FastAPI, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from typing import List
import better_tensorflow as btf
import json
from data_manager import DataManager
from datetime import datetime
from database_manager import DatabaseManager
from tqdm import tqdm

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
                scores = []
                files = sorted(
                    [
                        f
                        for f in os.listdir()
                        if f.startswith("rbf") and f.endswith(".weights")
                    ]
                )
                for file in files:
                    os.rename(file, "w_rbf.weight")
                    pred = btf.predict_rbf(array, True, True)
                    scores.append(pred)
                prediction = 0
                for i in range(len(scores)):
                    if scores[i] == 1:
                        prediction = i
                        break
            case "mlp":
                array = btf.convert_matrix_to_array(d.tolist())
                prediction = btf.predict_mlp(array, [], True, True)
            case "ols":
                array = btf.convert_matrix_to_array(d.tolist())
                weights = btf.import_weights_ols()
                prediction = btf.predict_ols([array], weights)[0]
                prediction = int(round(prediction))
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
                    svm = btf.KernelSVM("rbf", 2.0, lr=0.1, lambda_svm=0.01)
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
    for file in os.listdir():
        if file.startswith("rbf_") and file.endswith(".weights"):
            os.remove(file)

    dataset, dataset_test = DataManager.load_dataset(DATASET_PATH, filter_cat)
    for i in range(len(dataset)):
        dataset_split = [[], []]
        dataset__split_test = [[], []]

        for j in range(len(dataset)):
            if j == i:
                dataset_split[1].extend(dataset[j])
            else:
                dataset_split[0].extend(dataset[j])

        for j in range(len(dataset_test)):
            if j == i:
                dataset__split_test[1].extend(dataset_test[j])
            else:
                dataset__split_test[0].extend(dataset_test[j])

        n_pos = len(dataset_split[1])
        n_neg = len(dataset_split[0])
        if n_neg > n_pos:
            random.shuffle(dataset_split[0])
            dataset_split[0] = dataset_split[0][:n_pos]
        elif n_pos > n_neg:
            random.shuffle(dataset_split[1])
            dataset_split[1] = dataset_split[1][:n_neg]

        now = datetime.now()
        r = btf.train_rbf(
            dataset_split,
            dataset__split_test,
            [],
            number_clusters,
            gamma,
            True,
            True,
            f"train/mlp_{now.strftime('%Y-%m-%d_%H-%M-%S')}",
        )
        print(r)
        os.rename(
            "w_rbf.weight",
            f"rbf_{i}.weights",
        )

    return {"training": "OK"}


@app.post("/predict_svm")
async def predict_svm(file: UploadFile):
    with open("temp.mp3", "wb") as f:
        f.write(await file.read())

    with open("dataset.txt", "r") as f:
        cat = json.load(f)

    data = DataManager.load_data("temp.mp3")
    results = [0 for _ in range(len(cat))]
    svm = btf.KernelSVM("rbf", 2.0, lr=0.1, lambda_svm=0.01)

    for d in data:
        array = btf.convert_matrix_to_array(d.tolist())
        scores = []
        files = sorted(
            [f for f in os.listdir() if f.startswith("svm_") and f.endswith(".weights")]
        )
        for file in files:
            svm.load_weights_from(file)
            pred = svm.predict([array])[0]
            scores.append(pred)

        for i in range(len(scores)):
            if scores[i] == 1:
                results[i] += 1

    return {"prediction": cat[results.index(max(results))]}


@app.post("/predict_mlp")
async def predict_mlp(file: UploadFile):
    with open("temp.mp3", "wb") as f:
        f.write(await file.read())

    data = DataManager.load_data("temp.mp3")
    results = []
    for d in data:
        array = btf.convert_matrix_to_array(d.tolist())
        prediction = btf.predict_mlp(array, [], True, False)
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
    param: float,
    learning_rate: float,
    filter_cat: List[str],
    lambda_svm: float,
    kernel: str,
):
    import numpy as np

    dataset, dataset_test = DataManager.load_dataset(DATASET_PATH, filter_cat)

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

    for file in os.listdir():
        if file.startswith("svm_") and file.endswith(".weights"):
            os.remove(file)

    x_val = [item for sublist in dataset_test for item in sublist]
    y_val = [i for i, sublist in enumerate(dataset_test) for _ in sublist]

    svm = btf.KernelSVM(kernel, param, lr=learning_rate, lambda_svm=lambda_svm)
    for i in range(len(datasets_y)):
        x_data = np.array(
            [item for sublist in dataset for item in sublist], dtype=np.float64
        )
        y_data = np.array(datasets_y[i], dtype=np.float64)
        indices = np.arange(len(y_data))
        np.random.shuffle(indices)
        x_data = x_data[indices]
        y_data = y_data[indices]

        idx_pos = [idx for idx, val in enumerate(y_data) if val == 1]
        idx_neg = [idx for idx, val in enumerate(y_data) if val == -1]
        min_len = min(len(idx_pos), len(idx_neg))
        idx_pos = idx_pos[:min_len]
        idx_neg = idx_neg[:min_len]
        idx_balanced = idx_pos + idx_neg
        x_data_balanced = x_data[idx_balanced]
        y_data_balanced = y_data[idx_balanced]

        y_val_bin = [1 if label == i else -1 for label in y_val]

        svm.fit(
            x_data_balanced.tolist(),
            y_data_balanced.astype(int).tolist(),
            f"svm_{i}.weights",
            x_val,
            y_val_bin,
        )

    return {"training": "OK"}


@app.post("/train_ols")
async def training_ols(
    filter_cat: List[str],
    use_robust: bool = False,
):
    try:
        if not os.path.exists(DATASET_PATH):
            return {
                "training": "ERROR",
                "error": f"Le dossier {DATASET_PATH} n existe pas",
            }

        print(f"Chargement du dataset depuis: {DATASET_PATH}")
        print(f"Catégories filtrées: {filter_cat}")

        dataset, dataset_test = DataManager.load_dataset(DATASET_PATH, filter_cat)

        if not dataset:
            return {
                "training": "ERROR",
                "error": "Aucune donnée trouvée dans le dataset",
            }

        print(f"Dataset chargé: {len(dataset)} catégories")
        for i, cat in enumerate(dataset):
            print(f"Catégorie {i}: {len(cat)} échantillons")

        x_data = []
        y_data = []

        for i, category_data in enumerate(dataset):
            for sample in category_data:
                try:
                    if isinstance(sample, list) and len(sample) > 0:
                        if isinstance(sample[0], list):
                            array = btf.convert_matrix_to_array(sample)
                        else:
                            array = sample
                    else:
                        array = [sample] if not isinstance(sample, list) else sample

                    if array and len(array) > 0:
                        x_data.append(array)
                        y_data.append(float(i))
                    else:
                        print(f"Échantillon vide ignoré dans la catégorie {i}")

                except Exception as e:
                    print(
                        f"Erreur lors du traitement d'un échantillon de la catégorie {i}: {e}"
                    )
                    continue

        if not x_data or not y_data:
            return {
                "training": "ERROR",
                "error": "Aucune data valide trouvée après traitement",
            }

        print(f"data préparées: {len(x_data)} échantillons, {len(y_data)} labels")
        print(f"Dimensions des features: {len(x_data[0]) if x_data else 0}")

        if use_robust:
            print("Training with Robuste version ...")
            weights = btf.train_ols_robust(x_data, y_data)
        else:
            print("Training ith standard version...")
            weights = btf.train_ols(x_data, y_data)

        print(f"Entraînement terminé. Taille des poids: {len(weights)}")

        return {
            "training": "OK",
            "weights_size": len(weights),
            "samples_processed": len(x_data),
            "categories": len(dataset),
        }

    except Exception as e:
        print(f"Erreur durant l'entraînement OLS: {e}")
        import traceback

        traceback.print_exc()
        return {"training": "ERROR", "error": str(e)}


@app.post("/predict_ols")
async def predict_ols(file: UploadFile):
    with open("temp.mp3", "wb") as f:
        f.write(await file.read())

    data = DataManager.load_data("temp.mp3")
    results = []

    try:
        weights = btf.import_weights_ols()
    except Exception as e:
        return {"error": f"Impossible de charger les poids OLS: {str(e)}"}

    for d in data:
        array = btf.convert_matrix_to_array(d)
        prediction = btf.predict_ols([array], weights)[0]
        prediction_class = int(round(prediction))
        results.append(prediction_class)

    with open("dataset.txt", "r") as f:
        cat = json.load(f)

    most_common_prediction = max(set(results), key=results.count)

    if most_common_prediction >= len(cat):
        most_common_prediction = len(cat) - 1
    elif most_common_prediction < 0:
        most_common_prediction = 0

    print(f"OLS Results: {results}")
    print(f"Most common prediction: {most_common_prediction}")

    return {"prediction": cat[most_common_prediction]}


@app.post("/train_ols_multiclass")
async def training_ols_multiclass(
    filter_cat: List[str],
    use_robust: bool = False,
):
    dataset, dataset_test = DataManager.load_dataset(DATASET_PATH, filter_cat)
    print(f"Dataset train size: {sum(len(cat) for cat in dataset)}")
    print(f"Dataset test size: {sum(len(cat) for cat in dataset_test)}")

    x_data = []
    y_labels = []

    for i, category_data in enumerate(dataset):
        for sample in category_data:
            if (
                isinstance(sample, list)
                and len(sample) > 0
                and isinstance(sample[0], list)
            ):
                array = btf.convert_matrix_to_array(sample)
            else:
                array = sample if isinstance(sample, list) else [sample]

            x_data.append(array)
            y_labels.append(i)

    num_classes = len(dataset)

    for file in os.listdir():
        if file.startswith("ols_") and file.endswith(".weights"):
            os.remove(file)

    for class_idx in tqdm(range(num_classes)):
        y_binary = [1.0 if label == class_idx else 0.0 for label in y_labels]

        try:
            if use_robust:
                btf.train_ols_robust(x_data, y_binary)
            else:
                btf.train_ols(x_data, y_binary)
            os.rename("weights_ols.weights", f"ols_{class_idx}.weights")
            print("test")

        except Exception as e:
            return {
                "training": "ERROR",
                "error": f"Erreur pour la classe {class_idx}: {str(e)}",
            }

    return {"training": "OK", "num_classes": num_classes}


@app.post("/predict_ols_multiclass")
async def predict_ols_multiclass(file: UploadFile):
    with open("temp.mp3", "wb") as f:
        f.write(await file.read())

    data = DataManager.load_data("temp.mp3")
    results = []

    weight_files = sorted(
        [f for f in os.listdir() if f.startswith("ols_") and f.endswith(".weights")]
    )

    if not weight_files:
        return {
            "error": "Aucun modèle OLS trouvé. Veuillez d'abord entraîner le modèle."
        }

    for d in data:
        array = btf.convert_matrix_to_array(d)
        class_scores = []

        for weight_file in weight_files:
            try:
                weights = btf.import_weights_ols_from_file(weight_file)
                score = btf.predict_ols([array], weights)[0]
                class_scores.append(score)
            except Exception as e:
                print(f"Erreur avec {weight_file}: {e}")
                class_scores.append(0.0)

        predicted_class = class_scores.index(max(class_scores))
        results.append(predicted_class)

    with open("dataset.txt", "r") as f:
        cat = json.load(f)

    most_common_prediction = max(set(results), key=results.count)

    print(f"OLS Multiclass Results: {results}")
    print(f"Most common prediction: {most_common_prediction}")

    return {"prediction": cat[most_common_prediction]}


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
