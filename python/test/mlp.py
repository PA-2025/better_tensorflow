import better_tensorflow as btf
import os

dataset = [
    [[0, 0], [0, 1]],
    [[1, 0], [1, 1]],
]


file_name = "test"

btf.train_mlp(dataset, [], [], 300, [10, 10, 4], file_name, True, False, 0.01)

print(btf.predict_mlp([1, 1], [], True))
print(btf.predict_mlp([0, 1], [], True))
print(btf.predict_mlp([1, 0], [], True))
