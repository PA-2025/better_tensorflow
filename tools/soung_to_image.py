import sys
from typing import List

from pydub import AudioSegment
import matplotlib.pyplot as plt
from scipy.io import wavfile
from tempfile import mktemp
import os
from tqdm import tqdm

DURATION_MAX = 10


def split_audio(audio: AudioSegment, duration_max=DURATION_MAX) -> List[AudioSegment]:
    chunks = []
    start = 0
    while start < len(audio):
        end = min(start + duration_max * 1000, len(audio))
        chunks.append(audio[start:end])
        start = end
    return chunks


def sound_to_image():
    folders = os.listdir(sys.argv[1])
    for folder in tqdm(folders):
        files = os.listdir(sys.argv[1] + folder)
        for file in files:
            if not os.path.exists(f"{sys.argv[2]}/{folder}"):
                os.makedirs(f"{sys.argv[2]}/{folder}")

            mp3_audio = AudioSegment.from_file(
                f"{sys.argv[1]}/{folder}/{file}", format="mp3"
            )
            audio_chunks = split_audio(mp3_audio)

            for i, chunk in enumerate(audio_chunks):
                wname = mktemp(suffix=f"_{i}.wav")
                chunk.export(wname, format="wav")
                FS, data = wavfile.read(wname)
                if len(data.shape) > 1:
                    data = data.mean(axis=1)
                plt.specgram(data, Fs=FS)
                plt.axis("off")
                name = f"{file}-{i}"
                plt.savefig(
                    f"{sys.argv[2]}/{folder}/{name.replace('mp3', 'png')}",
                    format="png",
                    bbox_inches="tight",
                    pad_inches=0,
                )
                plt.close()


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python sound_to_image.py <input_folder> <output_folder>")
        sys.exit(1)
    if not os.path.exists(sys.argv[2]):
        os.makedirs(sys.argv[2])
    sound_to_image()
