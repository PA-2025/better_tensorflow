import better_tensorflow as btf
import os

"""dataset = [
    [[0, 0], [0, 1]],
    [[1, 0], [1, 1]],
]

file_name = "test"

btf.train_mlp(
    dataset, [], [], 1_000_000, [4], file_name, True, False, False, 0.01, 10000
)

print(btf.predict_mlp([1, 1], [], True, True))  # 1
print(btf.predict_mlp([0, 1], [], True, True))  # 0
print(btf.predict_mlp([0, 0], [], True, True))  # 0

dataset2 = [
    [[0, 0, 0], [0, 1, 0]],
    [[1, 0, 0], [1, 1, 0]],
    [[0, 0, 1], [0, 1, 1]],
    [[1, 0, 1], [1, 1, 1]],
]

file_name = "test2"

btf.train_mlp(
    dataset2, [], [], 1_000_000, [8, 4], file_name, True, False, False, 0.01, 10000
)

print(btf.predict_mlp([1, 1, 1], [], True, True))  # 3
print(btf.predict_mlp([0, 1, 1], [], True, True))  # 2"""

import numpy as np

X = np.array([[1], [2]])
Y = np.array([2, 4])

x_mlp = [[[20]], [[40]]]

btf.train_mlp(x_mlp, [], [20, 40], 100, [], "", False, False, False, 0.001, 10000)

print(btf.predict_mlp([1], [], False, True))  # 2
print(btf.predict_mlp([2], [], False, True))  # 3
