import better_tensorflow as btf

# Data
x = [[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]]
y = [5.0, 8.0, 11.0]

# Train OLS model
weights = btf.train_ols(x, y)
print("Learned weights:", weights)

# Predict
preds = btf.predict_ols(x, weights)
print("Predictions:", preds)
