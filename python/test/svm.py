from better_tensorflow import KernelSVM

# === Données XOR ===
X_xor = [
    [0.0, 0.0],
    [0.0, 1.0],
    [1.0, 0.0],
    [1.0, 1.0],
]
y_xor = [-1, 1, 1, -1]  # Sortie XOR encodée pour SVM

X_test = X_xor

svm_rbf = KernelSVM("rbf", 2.0, lr=0.1, lambda_svm=0.01, epochs=200)
svm_rbf.fit(X_xor, y_xor)
preds_rbf = svm_rbf.predict(X_test)

for x, pred in zip(X_test, preds_rbf):
    print(f"Entrée: {x} => Prédiction: {pred}")

svm_poly = KernelSVM("poly", 2, lr=0.1, lambda_svm=0.01, epochs=200)
svm_poly.fit(X_xor, y_xor)
preds_poly = svm_poly.predict(X_test)

for x, pred in zip(X_test, preds_poly):
    print(f"Entree: {x} => Prédiction: {pred}")
