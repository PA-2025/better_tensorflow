import os
from better_tensorflow import train_linear, predict_linear, load_linear_weights

# Entraînement
x_train = [0.0, 1.0, 2.0, 3.0, 4.0]
y_train = [0.0, 2.0, 4.0, 6.0, 8.0]

train_linear(
    x_data=x_train,
    y_data=y_train,
    mode="regression",
    verbose=True,
    epochs=1000,
    training_name="test_model"
)

# Vérification de l'existence du fichier de poids avant chargement
if os.path.exists("w_linear.weight"):
    m, b = load_linear_weights()

    # Prédiction
    x_test = [1.5, 2.5, 3.5]
    y_pred = predict_linear(x_test, m, b, mode="regression")

    print("Prédictions :", y_pred)
else:
    print("Fichier w_linear.weight introuvable. Veuillez vérifier que l'entraînement a bien exporté les poids.")
