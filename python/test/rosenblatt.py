from datetime import datetime
import os
import rosenblatt_py  # nom du module compilé PyO3

# Données exemples (régression ou classification)
x_data = [4.0, 5.0, 6.0]
y_data = [4.0, 5.0, 6.0]

now = datetime.now()
os.makedirs("train", exist_ok=True)
training_name = f"train/rosenblatt_{now.strftime('%Y-%m-%d_%H-%M-%S')}"

# Crée un fichier vide juste pour l'exemple (comme dans ton snippet)
open(training_name, "a").close()

# Entraîner le modèle Rosenblatt (mode regression ici)
rosenblatt_py.train_rosenblatt_py(
    x_data,
    y_data,
    "regression",
    True,
    100000,
    training_name
)

# Faire une prédiction avec le modèle entraîné (on passe None pour utiliser poids/biais sauvegardés)
predictions = rosenblatt_py.predict_rosenblatt_py(x_data, None, None, "regression")
print("Prédictions :", predictions)
