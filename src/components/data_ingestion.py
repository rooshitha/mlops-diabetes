from src.logger import logger
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split


class DataIngestion:
    """
    Handles loading and splitting the dataset.
    """

    def __init__(self):
        print("Data Ingestion Started...")

    def initiate_data_ingestion(self):

        logger.info("Loading Diabetes Dataset")

        diabetes = load_diabetes()

        X = diabetes.data
        y = diabetes.target

        print("Dataset Loaded Successfully!")
        print(f"Features Shape: {X.shape}")
        print(f"Target Shape: {y.shape}")

        logger.info("Dataset Loaded Successfully")

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=42
        )

        logger.info("Train-Test Split Completed")

        print("Train-Test Split Completed!")

        return X_train, X_test, y_train, y_test