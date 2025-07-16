import json
import os
import sys
from typing import List

import cv2
import numpy as np
from matplotlib import pyplot as plt
from pydub import AudioSegment

from data_manager import DataManager
import better_tensorflow as btf
from scipy.io import wavfile
from tempfile import mktemp
import onnxruntime as ort

import shutil


class TestE2E:
    @staticmethod
    def get_file_path(dataset_path: str) -> List[List[str]]:
        categories = os.listdir(dataset_path)
        file_paths = []
        for category in categories:
            category_path = os.path.join(dataset_path, category)
            files = os.listdir(category_path)
            for file in files:
                file_paths.append([os.path.join(category_path, file), category])
        return file_paths

    @staticmethod
    def get_cat() -> List[str]:
        f = open("dataset.txt", "r")
        cat = json.loads(f.read())
        f.close()
        return cat

    def run_test_mlp(self) -> int:
        dataset_test = self.get_file_path(sys.argv[1])
        cat = self.get_cat()
        score_e2e = 0
        for dt in dataset_test:
            data = DataManager.load_data(dt[0])
            results = []
            for d in data:
                array = btf.convert_matrix_to_array(d.tolist())
                prediction = btf.predict_mlp(array, [], True, False)
                results.append(prediction)

            print(
                f"Prediction for {dt[0]}: {results} -> {max(set(results), key=results.count)}"
            )
            if cat[int(max(set(results), key=results.count))] == dt[1]:
                score_e2e += 1
        print(f"Test MLP: {score_e2e}/{len(dataset_test)} correct predictions.")
        return score_e2e

    def run_resnet_test(self) -> int:
        dataset_test = self.get_file_path(sys.argv[1])
        cat = self.get_cat()
        score_e2e = 0
        ort_session = ort.InferenceSession("resnet_model_test.onnx")
        for dt in dataset_test:
            mp3_audio = AudioSegment.from_file(dt[0], format="mp3")
            audio_chunk = DataManager.split_audio(mp3_audio)
            results = []
            for audio in audio_chunk:
                wname = mktemp(".wav")
                audio.export(wname, format="wav")
                FS, data = wavfile.read(wname)
                if len(data.shape) > 1:
                    data = data.mean(axis=1)
                plt.specgram(data, Fs=FS)
                plt.axis("off")
                plt.savefig("temp.png", format="png", bbox_inches="tight", pad_inches=0)
                plt.close()
                mel_spectrogram = cv2.imread("temp.png")

                image_resized = cv2.resize(mel_spectrogram, (224, 224))
                image_rgb = cv2.cvtColor(image_resized, cv2.COLOR_BGR2RGB)
                image_tensor = image_rgb.astype(np.float32) / 255.0
                image_tensor = (
                    image_tensor - np.array([0.485, 0.456, 0.406])
                ) / np.array([0.229, 0.224, 0.225])
                image_tensor = np.transpose(image_tensor, (2, 0, 1))
                image_tensor = np.expand_dims(image_tensor, axis=0).astype(np.float32)

                outputs = ort_session.run(
                    ["predictions/Softmax"], {"input_1": image_tensor}
                )
                pred = np.argmax(outputs[0], axis=1)[0]
                results.append(pred)

            print(
                f"Prediction for {dt[0]}: {results} -> {max(set(results), key=results.count)}"
            )
            if cat[int(max(set(results), key=results.count))] == dt[1]:
                score_e2e += 1
        print(f"Test MLP: {score_e2e}/{len(dataset_test)} correct predictions.")
        return score_e2e

    def test_mlp_binary(self):
        weights = [
            "classical_hip-hop_jazz_mlp.weight",
            "pop_rock_mlp.weight",
            "techno_wajnberg_mlp.weight",
        ]
        dataset_test = self.get_file_path(sys.argv[1])
        cat = self.get_cat()
        score_e2e = 0
        for dt in dataset_test:
            data = DataManager.load_data(dt[0])
            results = []
            for d in data:
                array = btf.convert_matrix_to_array(d.tolist())
                for i in range(len(weights)):
                    shutil.copy(weights[i], "w_mlp.weight")
                    prediction = btf.predict_mlp(array, [], True, False)
                    prediction = (
                        int(prediction) if i == 0 else int(prediction) + i * 2 + 1
                    )
                    results.append(prediction)

            print(
                f"Prediction for {dt[0]}: {results} -> {max(set(results), key=results.count)}"
            )
            if cat[int(max(set(results), key=results.count))] == dt[1]:
                score_e2e += 1
        print(f"Test MLP: {score_e2e}/{len(dataset_test)} correct predictions.")
        return score_e2e

    def test_svm(self):
        dataset_test = self.get_file_path(sys.argv[1])
        svm = btf.KernelSVM("poly", 2.0, lr=0.1, lambda_svm=0.01, epochs=200)
        cat = self.get_cat()
        score_e2e = 0
        for dt in dataset_test:
            data = DataManager.load_data(dt[0])
            results = []
            for d in data:
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
                    svm.load_weights_from(file)
                    pred = svm.predict([array])[0]
                    scores.append(pred)
                print(scores)
                prediction = 0
                for i in range(len(scores)):
                    if scores[i] == 1:
                        prediction = i
                        break

                results.append(prediction)

            print(
                f"Prediction for {dt[0]}: {results} -> {max(set(results), key=results.count)}"
            )
            if cat[int(max(set(results), key=results.count))] == dt[1]:
                score_e2e += 1
        print(f"Test MLP: {score_e2e}/{len(dataset_test)} correct predictions.")
        return score_e2e

    def run_test_rbf(self) -> int:
        dataset_test = self.get_file_path(sys.argv[1])
        cat = self.get_cat()
        score_e2e = 0
        for dt in dataset_test:
            data = DataManager.load_data(dt[0])
            results = []
            for d in data:
                array = btf.convert_matrix_to_array(d.tolist())
                prediction = btf.predict_rbf(array, True, True)
                results.append(prediction)

            print(
                f"Prediction for {dt[0]}: {results} -> {max(set(results), key=results.count)}"
            )
            if cat[int(max(set(results), key=results.count))] == dt[1]:
                score_e2e += 1
        print(f"Test MLP: {score_e2e}/{len(dataset_test)} correct predictions.")
        return score_e2e


if __name__ == "__main__":
    test_e2e = TestE2E()
    # score = test_e2e.run_test_mlp()
    # print(f"Final Score: {score}")
    # score = test_e2e.run_resnet_test()
    # print(f"Final ResNet Score: {score}")
    # score = test_e2e.test_mlp_binary()
    # print(f"Final MLP Binary Score: {score}")
    # score = test_e2e.test_svm()
    # print(f"Final SVM Score: {score}")
    score = test_e2e.run_test_rbf()
    print(f"Final RBF Score: {score}")
