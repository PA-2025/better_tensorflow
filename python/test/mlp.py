import better_tensorflow as btf
import os

dataset = [
    [[[1.7]], [[2.2]], [[3.8]], [[4.6]], [[5.2]]],
    [[[6.5]], [[7.1]], [[8.3]], [[9.2]], [[10.0]]],
    [[[-1.5]], [[-0.2]], [[-2.8]], [[-9.6]], [[-5.2]]],
]


file_name = "test.txt"
if not os.path.exists("file_name"):
    f = open(file_name, "w")
    f.close()

btf.train_mlp(dataset, [], 1_000_000, [4], file_name, True, False)

print(btf.predict_mlp([[1.2]], True))
print(btf.predict_mlp([[10.1]], True))
print(btf.predict_mlp([[-10.1]], True))
