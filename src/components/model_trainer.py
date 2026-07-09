from src.logger import logger
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import os
import mlflow


class ModelTrainer:
    """
    Trains the machine learning model and saves it.
    """

    def __init__(self):
        print("Model Trainer Started...")

    def initiate_model_trainer(self, X_train, X_test, y_train, y_test):

        logger.info("Model Training Started")

        with mlflow.start_run():

            model = RandomForestRegressor(
                n_estimators=100,
                random_state=42
            )

            model.fit(X_train, y_train)

            predictions = model.predict(X_test)

            mse = mean_squared_error(y_test, predictions)
            r2 = r2_score(y_test, predictions)

            print("\nModel Evaluation")
            print("-----------------------")
            print(f"MSE : {mse:.2f}")
            print(f"R2  : {r2:.4f}")

            logger.info(f"MSE : {mse:.2f}")
            logger.info(f"R2 Score : {r2:.4f}")

            # Log parameters
            mlflow.log_param("model", "RandomForestRegressor")
            mlflow.log_param("n_estimators", 100)
            mlflow.log_param("random_state", 42)

            # Log metrics
            mlflow.log_metric("MSE", mse)
            mlflow.log_metric("R2 Score", r2)

            os.makedirs("artifacts", exist_ok=True)

            joblib.dump(model, "artifacts/model.pkl")

            # Log model file
            mlflow.log_artifact("artifacts/model.pkl")

            logger.info("Model Saved Successfully")

            print("\nModel Saved Successfully!")