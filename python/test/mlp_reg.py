import better_tensorflow as btf
import os

dataset_input = [
    [
        [1.7],
    ],
    [
        [6.5],
    ],
]

dataset_output = [40]

file_name = "test.txt"
if not os.path.exists("file_name"):
    f = open(file_name, "w")
    f.close()

btf.train_mlp(dataset_input, dataset_output, 1_000_000, [4,4], file_name, False, False, 0.01)

print(btf.predict_mlp([1.2], False))
print(btf.predict_mlp([6.5], False))
