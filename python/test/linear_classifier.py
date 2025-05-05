from better_tensorflow import train_logistic_regression

# Données d'exemple
x_data = [0.0, 1.0, 2.0, 3.0, 4.0]
y_data = [0.0, 0.0, 0.0, 1.0, 1.0]

train_logistic_regression(
    x_data=x_data, y_data=y_data, verbose=True, epochs=1000, training_name="test_model"
)
