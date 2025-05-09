import better_tensorflow as btf
import os

dataset = [
    [[0, 0], [0, 1]],
    [[1, 0], [1, 1]],
]

file_name = "test"

btf.train_mlp(dataset, [], [], 1_000_000, [4], file_name, True, False, False, 0.01)

print(btf.predict_mlp([1, 1], [], True))  # 1
print(btf.predict_mlp([0, 1], [], True))  # 0
print(btf.predict_mlp([0, 0], [], True))  # 0

dataset2 = [
    [[0, 0, 0], [0, 1, 0]],
    [[1, 0, 0], [1, 1, 0]],
    [[0, 0, 1], [0, 1, 1]],
    [[1, 0, 1], [1, 1, 1]],
]

file_name = "test2"

btf.train_mlp(dataset2, [], [], 1_000_000, [8, 4], file_name, True, False, False, 0.01)

print(btf.predict_mlp([1, 1, 1], [], True))  # 3
print(btf.predict_mlp([0, 1, 1], [], True))  # 2
