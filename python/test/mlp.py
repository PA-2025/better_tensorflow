import better_tensorflow as btf
import os

dataset = [
    [[0, 0], [0, 1]],
    [[1, 0], [1, 1]],
]


file_name = "test.txt"
if not os.path.exists("file_name"):
    f = open(file_name, "w")
    f.close()

btf.train_mlp(dataset, [], [], 1_000_000, [2, 7, 4], file_name, True, False, 0.01)

print(btf.predict_mlp([1, 1], [], True))
print(btf.predict_mlp([0, 1], [], True))
print(btf.predict_mlp([1, 0], [], True))
