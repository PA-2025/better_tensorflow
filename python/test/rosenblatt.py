import better_tensorflow as bt
from datetime import datetime
import os

# EX de dataset
x_data = [4.0, 5.0, 6.0]
y_data = [4.0, 5.0, 6.0]

# nom = (date + heure)
os.makedirs("train", exist_ok=True)
training_name = f"train/rosenblatt_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

print(f"Début de l'entraînement avec le fichier : {training_name}")

# Training
loss = bt.train_rosenblatt_py(
    x_data=x_data,
    y_data=y_data,
    mode="regression",
    save_in_db=True,
    training_name=training_name,
)
print(f"Entraînement terminé avec une perte de : {loss:.6f}")

# Prediction (les poids seront importés auto)
preds = bt.predict_rosenblatt_py(x_data, mode="regression")
print("Prédictions obtenues :")
for i, (x, pred, y) in enumerate(zip(x_data, preds, y_data), 1):
    print(f"  Point {i} : Entrée={x}, Prédiction={pred:.4f}, Réel={y}")

# Evaluation du modele : calcul de l'erreur absolue moyenne
errors = [abs(pred - y) for pred, y in zip(preds, y_data)]
mean_error = sum(errors) / len(errors)
print(f"Erreur absolue moyenne : {mean_error:.6f}")
