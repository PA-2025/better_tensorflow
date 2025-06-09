import numpy as np
from better_tensorflow import LinearSVM

def main():
    X = [
        [2, 3],
        [1, 1],
        [2, 1],
        [5, 7],
        [6, 8],
        [7, 7]
    ]
    y = [-1, -1, -1, 1, 1, 1]

    # Crée un modèle SVM linéaire
    svm = LinearSVM(lr=0.01, epochs=100, svm_lambda=0.01)

    # Entraîne le modèle
    svm.fit(X, y)


    X_test = [
        [2, 3],
        [1, 1],
        [2, 1],
        [5, 7],
        [6, 8],
        [7, 7]
    ]

    predictions = svm.predict(X_test)
    print("Prédictions :", predictions)

if __name__ == "__main__":
    main()
