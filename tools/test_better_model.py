import os

import numpy as np
import pandas as pd
import sys
import better_tensorflow as btf
from datetime import datetime
from tqdm import tqdm
from python.api.data_manager import DataManager


class Main:
    def __init__(self):
        self.results = pd.DataFrame(columns=["model", "image_size", "lr", "accuracy"])
        self.nb_test_to_save_results = int(sys.argv[1])
        self.test_layers = [
            [300, 200, 100],
        ]
        self.learning_rates = [0.01]
        self.dataset, self.dataset_test = [[], []]

    def run(self):
        test = [
            ["rock", "classical"],
            ["rock", "jazz"],
            ["rock", "wajnberg"],
            ["classical", "wajnberg"],
            ["jazz", "wajnberg"],
            ["techno", "rock"],
            ["techno", "classical"],
            ["techno", "jazz"],
            ["classical", "techno"],
            ["classical", "wajnberg"],
            ["techno", "rock"],

        ]
        for t in test:
            self.dataset, self.dataset_test = DataManager.load_dataset(
                "python/api/data/music_spec/", filter_categories=t
            )
            r = btf.train_mlp(
                self.dataset,
                self.dataset_test,
                [],
                int(sys.argv[2]),
                self.test_layers[0],
                "test_mlp",
                True,
                False,
                False,
                learning_rate=0.001,
                nb_epoch_to_save=int(sys.argv[2]),
            )
            # rename w_mlp.weight to w_mlp_rock_classical
            os.rename(
                "w_mlp.weight",
                f"test_mlp_w_mlp_{t[0]}_{t[1]}.weight",
            )
            print(t)
            print(r)


if __name__ == "__main__":
    main = Main()
    main.run()
    main.results.to_csv("results.csv", index=False)
