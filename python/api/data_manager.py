import better_tensorflow as btf
import os
import json
from tqdm import tqdm
from typing import Optional, List, Tuple
from pydub import AudioSegment
import matplotlib.pyplot as plt
from scipy.io import wavfile
from tempfile import mktemp
import cv2
import random


class DataManager:
    @staticmethod
    def load_dataset(
        dataset_path: str,
        filter_categories: Optional[List[str]] = None,
        split: Optional[float] = 0.1,
    ) -> Tuple[List[List], List[List]]:
        dataset = []
        folders = os.listdir(dataset_path)
        f = open("dataset.txt", "w")
        json.dump(folders, f)
        f.close()
        for folder in tqdm(folders):
            if filter_categories and folder not in filter_categories:
                print(f"Skipping {folder}...")
                continue
            files = os.listdir(dataset_path + folder)
            cat_dataset = []
            for file in files:
                mel_spectrogram = cv2.imread(
                    f"{dataset_path}/{folder}/{file}",
                )
                mel_spectrogram = btf.convert_image_to_array(mel_spectrogram)
                cat_dataset.append(mel_spectrogram)
            dataset.append(cat_dataset)
        return DataManager.split_dataset(dataset, split)

    @staticmethod
    def split_dataset(
        dataset: List[List], split: float
    ) -> Tuple[List[List], List[List]]:
        new_form_dataset = []
        for index_cat in range(len(dataset)):
            for data in dataset[index_cat]:
                new_form_dataset.append([data, index_cat])
        random.shuffle(new_form_dataset)
        dataset_train = [[] for _ in range(len(dataset))]
        dataset_test = [[] for _ in range(len(dataset))]
        print(
            f"Dataset train size:{len(new_form_dataset) - int(len(new_form_dataset) * split)}"
        )
        print(f"Dataset test size:{int(len(new_form_dataset) * split)}")
        for data in new_form_dataset[: int(len(new_form_dataset) * split)]:
            dataset_test[data[1]].append(data[0])
        for data in new_form_dataset[int(len(new_form_dataset) * split) :]:
            dataset_train[data[1]].append(data[0])
        return dataset_train, dataset_test

    @staticmethod
    def load_data(data_path: str):
        mp3_audio = AudioSegment.from_file(data_path, format="mp3")
        wname = mktemp(".wav")
        mp3_audio.export(wname, format="wav")
        FS, data = wavfile.read(wname)
        if len(data.shape) > 1:
            data = data.mean(axis=1)
        plt.specgram(data, Fs=FS)
        plt.axis("off")
        plt.savefig("temp.png", format="png", bbox_inches="tight", pad_inches=0)
        plt.close()
        mel_spectrogram = cv2.imread("temp.png")
        return mel_spectrogram

    @staticmethod
    def find_dataset_categories(dataset_path: str) -> List[str]:
        categories = []
        folders = os.listdir(dataset_path)
        for folder in folders:
            if os.path.isdir(f"{dataset_path}/{folder}"):
                categories.append(folder)

        return categories
