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
            [64, 32, 16, 8, 4],
            [128, 64, 32, 16, 8],
            [256, 128, 64, 32, 16],
        ]
        self.learning_rates = [0.01, 0.001]
        self.dataset, self.dataset_test = DataManager.load_dataset(
            "python/api/data/music_spec/", filter_categories=["techno", "wajnberg"]
        )

    def run(self):
        for layers in tqdm(self.test_layers):
            for lr in self.learning_rates:
                result = []
                for dataset, dataset_test, name in [
                    (self.dataset, self.dataset_test, "(50,37)"),
                    # (self.dataset_little, self.dataset_test_little, "(10*7)"),
                ]:
                    for i in range(self.nb_test_to_save_results):
                        now = datetime.now()

                        r = btf.train_mlp(
                            dataset,
                            dataset_test,
                            [],
                            int(sys.argv[2]),
                            layers,
                            f"train/mlp_{now.strftime('%Y-%m-%d_%H-%M-%S')}",
                            True,
                            False,
                            False,
                            learning_rate=lr,
                            nb_epoch_to_save=int(sys.argv[2]),
                        )
                        result.append(r)

                        print(
                            f"Test {i + 1}/{self.nb_test_to_save_results} for model mlp-{layers}-lr-{lr}-size-{name}"
                        )
                        print(f"Result: {r}")

                    self.results = pd.concat(
                        [
                            self.results,
                            pd.DataFrame(
                                [
                                    {
                                        "model": f"mlp-{layers}-mean-{self.nb_test_to_save_results}",
                                        "image_size": name,
                                        "lr": lr,
                                        "accuracy": np.max(result),
                                    }
                                ]
                            ),
                        ],
                        ignore_index=True,
                    )


if __name__ == "__main__":
    main = Main()
    main.run()
    main.results.to_csv("results.csv", index=False)
