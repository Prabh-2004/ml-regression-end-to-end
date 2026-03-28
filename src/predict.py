import joblib
import pandas as pd

# Paths
MODEL_PATH = "models/model.pkl"
PREPROCESSOR_PATH = "models/preprocessor.pkl"

class HousePricePredictor:
    def __init__(self):
        print("Loading model and preprocessor...")
        self.model = joblib.load(MODEL_PATH)
        self.preprocessor = joblib.load(PREPROCESSOR_PATH)

    def predict(self, input_data: dict):
        """
        input_data: dictionary with feature values
        """

        # Convert input to DataFrame
        input_df = pd.DataFrame([input_data])

        # Apply preprocessing
        processed_data = self.preprocessor.transform(input_df)

        # Make prediction
        prediction = self.model.predict(processed_data)

        return float(prediction[0])


# For testing locally
if __name__ == "__main__":
    predictor = HousePricePredictor()

    sample_input = {
        "area": 1200,
        "bedrooms": 3,
        "bathrooms": 2,
        "stories": 2,
        "parking": 1,
        "furnishingstatus": "furnished"
    }

    result = predictor.predict(sample_input)
    print(f"Predicted Price: {result}")