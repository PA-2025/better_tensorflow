import better_tensorflow as btf

dataset = [
    [[0, 0]],
    [[1, 1]],
]

btf.train_rbf(dataset, [], 1_000_000, 4, "file_name", True, False, False, 0.01, 10000)

print(btf.predict_rbf([1, 1], True, True))  # 1
print(btf.predict_rbf([0, 0], True, True))  # 0
