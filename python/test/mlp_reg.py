import better_tensorflow as btf
import os

dataset_input = [
    [
        [1],
    ],
    [
        [2],
    ],
]

dataset_output = [10, 20]

file_name = "test"

btf.train_mlp(
    dataset_input,
    [],
    dataset_output,
    1_000_000,
    [4],
    file_name,
    False,
    False,
    False,
    0.01,
    10000,
)

print(btf.predict_mlp([1], [], False, True))
print(btf.predict_mlp([2], [], False, True))
print(btf.predict_mlp([3], [], False, True))
