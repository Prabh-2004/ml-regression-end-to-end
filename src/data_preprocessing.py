import os
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Paths
RAW_DATA_PATH = "./data/raw/housing_data.csv"
PROCESSED_DIR = "./data/processed"
PREPROCESSOR_PATH = "models/preprocessor.pkl"


def load_data(path=RAW_DATA_PATH):
    df = pd.read_csv(path)

    # Drop unwanted index column if present
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])

    return df


def main():
    print("Loading Data...")
    df = load_data()

    target = "price"

    # Define X and y
    X = df.drop(columns=[target])
    y = df[target]

    print("Splitting Data...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=24
    )

    # Identify column types
    numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
    categorical_features = X.select_dtypes(include=['object']).columns

    # Pipelines
    numeric_pipeline = Pipeline(steps=[
        ("pass", "passthrough")
    ])

    categorical_pipeline = Pipeline(steps=[
        ("encoder", OrdinalEncoder(
            handle_unknown="use_encoded_value",
            unknown_value=-1
        ))
    ])

    # Column Transformer
    preprocessor = ColumnTransformer(transformers=[
        ("num", numeric_pipeline, numeric_features),
        ("cat", categorical_pipeline, categorical_features)
    ])

    print("Fitting Preprocessor on TRAIN data...")
    X_train_processed = preprocessor.fit_transform(X_train)

    print("Transforming TEST data...")
    X_test_processed = preprocessor.transform(X_test)

    # Save artifacts
    print("Saving artifacts...")
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    os.makedirs("models", exist_ok=True)

    joblib.dump(preprocessor, PREPROCESSOR_PATH)

    # Save processed data
    joblib.dump(X_train_processed, os.path.join(PROCESSED_DIR, "X_train.pkl"))
    joblib.dump(X_test_processed, os.path.join(PROCESSED_DIR, "X_test.pkl"))
    joblib.dump(y_train, os.path.join(PROCESSED_DIR, "y_train.pkl"))
    joblib.dump(y_test, os.path.join(PROCESSED_DIR, "y_test.pkl"))

    print("Data preprocessing completed successfully!")


if __name__ == "__main__":
    main()