import better_tensorflow as btf

# X = [[1, 2], [2, 3], [3, 4]], Y = [5, 8, 11]
x = [[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]]
y = [5.0, 8.0, 11.0]

weights = btf.train_ols(x, y)
print("Poids appris:", weights)

# Prédiction
preds = btf.predict_ols(x, weights)
print("Prédictions:", preds)
