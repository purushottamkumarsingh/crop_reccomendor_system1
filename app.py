import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split
from src.components.model_trainer import ModelTrainer, CustomException, logger

def main():
    try:
        logger.info("Starting the ML application...")

        # ----------------- Load Data -----------------
        # Replace this with your CSV or dataset
        data_path = os.path.join("data", "dataset.csv")
        if not os.path.exists(data_path):
            logger.warning(f"{data_path} not found. Using Iris dataset as default.")
            from sklearn.datasets import load_iris
            data = load_iris(as_frame=True)
            X = data.data
            y = data.target
        else:
            df = pd.read_csv(data_path)
            X = df.drop("target", axis=1)
            y = df["target"]

        # ----------------- Split Data -----------------
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        logger.info("Data split into train and test sets.")

        # ----------------- Train Model -----------------
        trainer = ModelTrainer()
        model = trainer.initiate_model_trainer(X_train, y_train)

        # ----------------- Evaluate Model -----------------
        accuracy = trainer.evaluate_model(model, X_test, y_test)
        logger.info(f"Final Model Accuracy: {accuracy}")

        logger.info("ML application finished successfully.")

    except Exception as e:
        raise CustomException(e, sys)

if __name__ == "__main__":
    main()
