import librosa
from pythin import List
import os
class DataManager:


    @staticmethod
    def load_dataset(dataset_path: str) -> List[List[]]:
        dataset = []
        folders = os.listdir(dataset_path)
        for folder in folders:
            files = os.listdir(dataset_path + folder)
            cat_dataset = []
            for file in files :
                audio_data, sample_rate = librosa.load(f"{dataset_path}/{folder}/{file}", sr=None)
                mel_spectrogram = librosa.feature.melspectrogram(y=audio_data, sr=sample_rate)
                cat_dataset.append(mel_spectrogram)
            dataset.append(cat_dataset)
        return dataset
