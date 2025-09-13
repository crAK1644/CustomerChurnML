## model assumptions
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import joblib
import os
from datetime import datetime
from sklearn.model_selection import train_test_split

# Add the src directory to the Python path
sys.path.append(str(Path(__file__).parent.parent))

from data_preprocess.data_preprocesser import DataPreprocesser
from data_loader.data_loader import DataLoader


class ChurnModel:
    def __init__(self, data_file_path):
        self.data_loader = DataLoader(data_file_path)
        self.preprocesser = DataPreprocesser()
        self.model = None

    def load_and_preprocess_data(self):
        # Load data
        df = self.data_loader.load_data()

        # Clean the data
        # Convert TotalCharges to numeric (it might be string with spaces)
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

        # Convert Yes/No to 1/0 for binary columns
        binary_cols = ["Partner", "Dependents", "PhoneService", "PaperlessBilling"]
        for col in binary_cols:
            if col in df.columns:
                df[col] = df[col].map({"Yes": 1, "No": 0})

        # Drop customerID as it's not useful for prediction
        if "customerID" in df.columns:
            df = df.drop(columns=["customerID"])

        # Separate features and target
        X = df.drop(columns=["Churn"])
        y = df["Churn"]

        # Create the preprocessing pipeline
        self.preprocesser.create_pipeline()

        # Preprocess features using fit_transform
        X_preprocessed = self.preprocesser.fit_transform(X)

        # Convert sparse matrix to dense array if needed
        try:
            if hasattr(X_preprocessed, "toarray"):
                X_preprocessed = X_preprocessed.toarray()
        except:
            pass

        # Get feature names after preprocessing
        try:
            feature_names = self.preprocesser.get_feature_names()
            X_preprocessed_df = pd.DataFrame(
                X_preprocessed, columns=feature_names, index=X.index
            )
        except:
            # Fallback if feature names are not available
            X_preprocessed_df = pd.DataFrame(X_preprocessed, index=X.index)

        # Combine with target
        df_processed = X_preprocessed_df.copy()
        df_processed["Churn"] = y

        return df_processed

    def train_model(self, df, algorithm="decision_tree"):
        from sklearn.ensemble import (
            GradientBoostingClassifier,
            RandomForestClassifier,
            AdaBoostClassifier,
        )
        from sklearn.linear_model import LogisticRegression
        from sklearn.svm import SVC
        from sklearn.tree import DecisionTreeClassifier
        from sklearn.neighbors import KNeighborsClassifier

        # Split data into features and target
        X = df.drop(columns=["Churn"])
        y = df["Churn"]
        # Split into training and test sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # Define available algorithms
        algorithms = {
            "decision_tree": DecisionTreeClassifier(random_state=42),
            "random_forest": RandomForestClassifier(n_estimators=100, random_state=42),
            "gradient_boosting": GradientBoostingClassifier(random_state=42),
            "logistic_regression": LogisticRegression(random_state=42, max_iter=1000),
            "svm": SVC(random_state=42),
            "knn": KNeighborsClassifier(n_neighbors=5),
            "ada_boost": AdaBoostClassifier(random_state=42),
        }

        # Select and train the specified algorithm
        if algorithm not in algorithms:
            raise ValueError(
                f"Algorithm '{algorithm}' not supported. Available: {list(algorithms.keys())}"
            )

        self.model = algorithms[algorithm]
        self.algorithm_name = algorithm
        print(f"Training {algorithm.replace('_', ' ').title()} model...")
        self.model.fit(X_train, y_train)
        return X_test, y_test

    def compare_algorithms(self, df):
        """Compare multiple algorithms and return their performance scores"""
        from sklearn.metrics import (
            accuracy_score,
            precision_score,
            recall_score,
            f1_score,
        )

        algorithms = [
            "decision_tree",
            "random_forest",
            "gradient_boosting",
            "logistic_regression",
            "svm",
            "knn",
            "ada_boost",
        ]

        results = {}

        # Split data once for fair comparison
        X = df.drop(columns=["Churn"])
        y = df["Churn"]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        for algo in algorithms:
            print(f"\n{'=' * 50}")
            print(f"Training and evaluating: {algo.replace('_', ' ').title()}")
            print(f"{'=' * 50}")

            # Train model with current algorithm
            temp_model = self.__class__(self.data_loader.data_file_path)
            temp_model.preprocesser = self.preprocesser  # Use same preprocessor
            temp_model.train_single_algorithm(X_train, X_test, y_train, y_test, algo)

            # Get predictions and calculate metrics
            y_pred = temp_model.model.predict(X_test)

            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)

            results[algo] = {
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1_score": f1,
            }

            # Print detailed classification report
            temp_model.evaluate(X_test, y_test)

        return results

    def train_single_algorithm(self, X_train, X_test, y_train, y_test, algorithm):
        """Helper method for training a single algorithm"""
        from sklearn.ensemble import (
            GradientBoostingClassifier,
            RandomForestClassifier,
            AdaBoostClassifier,
        )
        from sklearn.linear_model import LogisticRegression
        from sklearn.svm import SVC
        from sklearn.tree import DecisionTreeClassifier
        from sklearn.neighbors import KNeighborsClassifier

        algorithms = {
            "decision_tree": DecisionTreeClassifier(random_state=42),
            "random_forest": RandomForestClassifier(n_estimators=100, random_state=42),
            "gradient_boosting": GradientBoostingClassifier(random_state=42),
            "logistic_regression": LogisticRegression(random_state=42, max_iter=1000),
            "svm": SVC(random_state=42),
            "knn": KNeighborsClassifier(n_neighbors=5),
            "ada_boost": AdaBoostClassifier(random_state=42),
        }

        self.model = algorithms[algorithm]
        self.algorithm_name = algorithm
        self.model.fit(X_train, y_train)

    def predict(self, X):
        if self.model is None:
            raise ValueError("Model has not been trained yet.")
        return self.model.predict(X)

    def evaluate(self, X_test, y_test):
        from sklearn.metrics import classification_report

        y_pred = self.predict(X_test)
        report = classification_report(y_test, y_pred)
        print(report)

    def save_model(self, model_dir="models", model_name=None):
        """
        Save the trained model and preprocessor to disk

        Args:
            model_dir (str): Directory to save the model
            model_name (str): Name for the model file. If None, uses timestamp
        """
        if self.model is None:
            raise ValueError("No model to save. Train the model first.")

        # Create models directory if it doesn't exist
        os.makedirs(model_dir, exist_ok=True)

        # Generate model name if not provided
        if model_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_name = f"churn_model_{timestamp}"

        # Save model and preprocessor
        model_path = os.path.join(model_dir, f"{model_name}.pkl")
        preprocessor_path = os.path.join(model_dir, f"{model_name}_preprocessor.pkl")

        joblib.dump(self.model, model_path)
        joblib.dump(self.preprocesser, preprocessor_path)

        print(f"Model saved to: {model_path}")
        print(f"Preprocessor saved to: {preprocessor_path}")

        return model_path, preprocessor_path

    def load_model(self, model_path, preprocessor_path):
        """
        Load a previously saved model and preprocessor

        Args:
            model_path (str): Path to the saved model file
            preprocessor_path (str): Path to the saved preprocessor file
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        if not os.path.exists(preprocessor_path):
            raise FileNotFoundError(f"Preprocessor file not found: {preprocessor_path}")

        self.model = joblib.load(model_path)
        self.preprocesser = joblib.load(preprocessor_path)

        print(f"Model loaded from: {model_path}")
        print(f"Preprocessor loaded from: {preprocessor_path}")

    @classmethod
    def load_trained_model(cls, model_path, preprocessor_path, data_file_path=None):
        """
        Class method to create a ChurnModel instance with a pre-trained model

        Args:
            model_path (str): Path to the saved model file
            preprocessor_path (str): Path to the saved preprocessor file
            data_file_path (str): Optional path to data file

        Returns:
            ChurnModel: Instance with loaded model and preprocessor
        """
        instance = cls(data_file_path)
        instance.load_model(model_path, preprocessor_path)
        return instance


if __name__ == "__main__":
    # Correct path to the CSV file
    data_file_path = "/Users/ayberkkarataban/MallCustomersML/Telco_Customer_Churn.csv"

    print("=" * 60)
    print("CUSTOMER CHURN PREDICTION - ALGORITHM COMPARISON")
    print("=" * 60)

    # Initialize model and preprocess data
    churn_model = ChurnModel(data_file_path)
    df = churn_model.load_and_preprocess_data()

    # Test different algorithms
    algorithms_to_test = [
        "decision_tree",
        "random_forest",
        "gradient_boosting",
        "logistic_regression",
        "knn",
    ]

    best_algorithm = None
    best_accuracy = 0
    results_summary = []

    for algorithm in algorithms_to_test:
        print(f"\n{'=' * 50}")
        print(f"Testing: {algorithm.replace('_', ' ').title()}")
        print(f"{'=' * 50}")

        # Create a fresh model instance for each algorithm
        model = ChurnModel(data_file_path)
        model.preprocesser = churn_model.preprocesser  # Use same preprocessor

        # Train with specific algorithm
        X_test, y_test = model.train_model(df, algorithm=algorithm)
        print("Model training complete.")

        # Evaluate
        model.evaluate(X_test, y_test)

        # Calculate accuracy for comparison
        from sklearn.metrics import accuracy_score

        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)

        results_summary.append(
            {"algorithm": algorithm, "accuracy": accuracy, "model": model}
        )

        # Track best performing algorithm
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_algorithm = algorithm
            best_model = model

    # Print summary
    print(f"\n{'=' * 60}")
    print("ALGORITHM COMPARISON SUMMARY")
    print(f"{'=' * 60}")
    for result in results_summary:
        print(
            f"{result['algorithm'].replace('_', ' ').title():20}: {result['accuracy']:.4f}"
        )

    if best_algorithm:
        print(
            f"\nBest performing algorithm: {best_algorithm.replace('_', ' ').title()} ({best_accuracy:.4f})"
        )

        # Save the best model
        try:
            model_path, preprocessor_path = best_model.save_model(
                model_name=f"best_churn_model_{best_algorithm}"
            )
            print(f"\nBest model ({best_algorithm}) saved successfully!")
            print(f"Model saved to: {model_path}")

        except Exception as e:
            print(f"Error saving model: {e}")
    else:
        print("\nNo models were successfully trained.")
