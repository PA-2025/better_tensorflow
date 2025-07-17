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

# === SVM RBF ===
svm_rbf = KernelSVM("rbf", 2.0, lr=0.1, lambda_svm=0.01)
print("=== SVM avec noyau RBF ===")
svm_rbf.fit(X_xor, y_xor, path="rbf_xor.weights")
print("=== Prédictions RBF ===")
preds_rbf = svm_rbf.predict(X_test)

for x, pred in zip(X_test, preds_rbf):
    print(f"Entrée: {x} => Prédiction RBF: {pred}")

# === SVM Polynomial ===
svm_poly = KernelSVM("poly", 2, lr=0.1, lambda_svm=0.01)
print("=== SVM avec noyau Polynomial ===")
svm_poly.fit(X_xor, y_xor, path="poly_xor.weights")
print("=== Prédictions Polynomial ===")
preds_poly = svm_poly.predict(X_test)

for x, pred in zip(X_test, preds_poly):
    print(f"Entrée: {x} => Prédiction Poly: {pred}")
