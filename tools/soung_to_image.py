import sys
from pydub import AudioSegment
import matplotlib.pyplot as plt
from scipy.io import wavfile
from tempfile import mktemp
import os
from tqdm import tqdm


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
            wname = mktemp(".wav")
            mp3_audio.export(wname, format="wav")
            FS, data = wavfile.read(wname)
            if len(data.shape) > 1:
                data = data.mean(axis=1)
            plt.specgram(data, Fs=FS)
            plt.axis("off")
            plt.savefig(
                f"{sys.argv[2]}/{folder}/{file.replace('mp3', 'png')}",
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
