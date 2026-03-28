import os
import numpy as np
import joblib
import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score

PROCESSED_DIR = "./data/processed"
MODEL_PATH = "./models/model.pkl"

def load_data():
    X_train = joblib.load(os.path.join(PROCESSED_DIR, "X_train.pkl"))
    X_test = joblib.load(os.path.join(PROCESSED_DIR, "X_test.pkl"))
    y_train = joblib.load(os.path.join(PROCESSED_DIR, "y_train.pkl"))
    y_test = joblib.load(os.path.join(PROCESSED_DIR, "y_test.pkl"))

    return X_train, X_test, y_train, y_test

def train_model(X_train, y_train):
    model = RandomForestRegressor(
        n_estimators=100, 
        max_depth=10, 
        random_state=24,
        n_jobs=-1
    )
    model.fit(X_train ,y_train)
    return model

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)

    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r_squared = r2_score(y_test, y_pred)

    return rmse, r_squared

def save_model(model):
    os.makedirs("models", exist_ok=True)
    joblib.dump(model, MODEL_PATH)

def main():
    # set an mlflow experiment
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    mlflow.set_experiment("house_price_prediction")

    with mlflow.start_run():
        print("Loading Data...")
        X_train, X_test, y_train, y_test = load_data()

        # define the infer signature
        # signature = infer_signature(X_train, y_train)

        print("Training Model...")
        model = train_model(X_train, y_train)

        print("Evaluate Model...")
        rmse, r2 = evaluate_model(model, X_test, y_test)

        # Log model parameters
        mlflow.log_param("n_estimators", 100)
        mlflow.log_param("max_depth", 10)
        # mlflow.log_param("learning_rate", 0.05)

        # Log the metrics
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2", r2)

        # log the model
        mlflow.sklearn.log_model(sk_model=model, name="regression_model")

        print(f"RMSE: {rmse}")
        print(f"R2 Score: {r2}")
    
    # Save the model
    save_model(model)

    print(f"The Model has been Trained.")

if __name__=="__main__":
    main()