import better_tensorflow as btf
import os
import json
from tqdm import tqdm
from typing import Optional, List
from pydub import AudioSegment
import matplotlib.pyplot as plt
from scipy.io import wavfile
from tempfile import mktemp
import cv2


class DataManager:
    @staticmethod
    def load_dataset(
        dataset_path: str,
        filter_categories: Optional[List[str]] = None,
    ) -> List[List]:
        dataset = []
        shape = 128 * 128
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
        return dataset

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
