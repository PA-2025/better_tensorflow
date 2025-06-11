import numpy as np
import pandas as pd
import sys
import better_tensorflow as btf
from datetime import datetime

from python.api.data_manager import DataManager


class Main:
    def __init__(self):
        self.results = pd.DataFrame(columns=["model", "accuracy"])
        self.nb_test_to_save_results = int(sys.argv[1])
        self.test_layers = [[4], [4, 4], [10], [4, 10]]
        self.learning_rates = [0.1, 0.001]
        self.dataset, self.dataset_test = DataManager.load_dataset(
            "python/api/data/music_spec/"
        )

    def run(self):
        for layers in self.test_layers:
            for lr in self.learning_rates:
                result = []
                for i in range(self.nb_test_to_save_results):
                    now = datetime.now()

                    r = btf.train_mlp(
                        self.dataset,
                        self.dataset_test,
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
                        f"Test {i + 1}/{self.nb_test_to_save_results} for model mlp-{layers[0]}-lr-{lr}"
                    )
                    print(f"Result: {r}")

                self.results = self.results.append(
                    {
                        "model": f"mlp-{layers[0]}-lr-{lr}-mean-{self.nb_test_to_save_results}",
                        "accuracy": np.max(result),
                    },
                    ignore_index=True,
                )


if __name__ == "__main__":
    main = Main()
    main.run()
    main.results.to_csv("results.csv", index=False)
