from src.components.data_ingestion import DataIngestion
from src.components.model_trainer import ModelTrainer


class TrainingPipeline:

    def start_training_pipeline(self):
        print("=" * 50)
        print("Training Pipeline Started")
        print("=" * 50)

        # Data Ingestion
        data_ingestion = DataIngestion()

        X_train, X_test, y_train, y_test = (
            data_ingestion.initiate_data_ingestion()
        )

        # Model Training
        model_trainer = ModelTrainer()

        model_trainer.initiate_model_trainer(
            X_train,
            X_test,
            y_train,
            y_test
        )

        print("=" * 50)
        print("Training Pipeline Completed")
        print("=" * 50)