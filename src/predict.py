import joblib
import numpy as np

# Load the trained model
model = joblib.load("../models/best_model.pkl")

# Define the prediction function
def predict_default(features):
    features = np.array(features).reshape(1, -1)
    prediction = model.predict(features)
    probability = model.predict_proba(features)[0][1]
    return prediction[0], probability

# Example: Replace this with actual input features
# IMPORTANT: Input should match the number of features the model was trained on
if __name__ == "__main__":
    # Example input (replace with real feature values in order)
    example_features = [20000, 2, 2, 1, 24, 2, 2, -1, -1, -2, -2,
                    3913, 3102, 689, 0, 0, 0,
                    0, 689, 0, 0, 0, 0]
    prediction, probability = predict_default(example_features)

    print(f"Prediction (1 = default, 0 = no default): {prediction}")
    print(f"Probability of default: {probability:.2f}")
