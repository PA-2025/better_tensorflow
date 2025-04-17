import better_tensorflow as btf
import os

dataset = [
    [
        [1.7],
    ],
    [
        [6.5],
    ],
    [
        [-10.1],
    ],
]


file_name = "test.txt"
if not os.path.exists("file_name"):
    f = open(file_name, "w")
    f.close()

btf.train_mlp(dataset, [], 1_000_000, [4], file_name, True, False, 0.01)

print(btf.predict_mlp([1.2], True))
print(btf.predict_mlp([10.1], True))
print(btf.predict_mlp([-10.1], True))
