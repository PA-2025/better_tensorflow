import better_tensorflow as btf

dataset = [
    [[0, 0]],
    [[20, 20]],
    [[40, 40]],
]

btf.train_rbf(dataset, [], [], 3, 1, True, False, "file_name")

print(btf.predict_rbf([0, 0], 1, True))  # 1
print(btf.predict_rbf([20, 20], 1, True))  # 0
