import librosa
from librosa.util import fix_length
from typing import List
import better_tensorflow as btf
import os
import json
from tqdm import tqdm


class DataManager:
    @staticmethod
    def load_dataset(dataset_path: str) -> List[List]:
        dataset = []
        shape = 128 * 128
        folders = os.listdir(dataset_path)
        f = open("dataset.txt", "w")
        json.dump(folders, f)
        f.close()
        for folder in tqdm(folders):
            files = os.listdir(dataset_path + folder)
            cat_dataset = []
            for file in files:
                audio_data, sample_rate = librosa.load(
                    f"{dataset_path}/{folder}/{file}", sr=None
                )
                mel_spectrogram = librosa.feature.melspectrogram(
                    y=audio_data, sr=sample_rate
                )
                mel_spectrogram = fix_length(mel_spectrogram, size=shape)
                mel_spectrogram = btf.convert_matrix_to_array(mel_spectrogram)
                cat_dataset.append(mel_spectrogram)
            dataset.append(cat_dataset)
        return dataset

    @staticmethod
    def load_data(data_path: str):
        audio_data, sample_rate = librosa.load(f"{data_path}", sr=None)
        mel_spectrogram = librosa.feature.melspectrogram(y=audio_data, sr=sample_rate)

        return mel_spectrogram
