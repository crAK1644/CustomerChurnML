### first model training and evaluation
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
# Import with robust fallbacks so this file can be run as a script or as a module
try:
    from customer_churn_ml.src.data_loader.data_loader import DataLoader
    from customer_churn_ml.src.data_preprocess.data_preprocesser import DataPreprocesser
except ModuleNotFoundError:
    # Fallback: add the package's src folder to sys.path then import
    import sys
    sys.path.append(str(Path(__file__).resolve().parents[1]))  # .../customer_churn_ml/src
    from data_loader.data_loader import DataLoader
    from data_preprocess.data_preprocesser import DataPreprocesser


class Model:
    def __init__(self, data_path, target_column: str = "Churn", config_path: str | Path | None = None):
        self.data_path = data_path
        self.model = LogisticRegression(max_iter=200)
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

        # Explicitly configure target column and preprocessing config path
        self.target_column = target_column
        default_cfg = Path(__file__).resolve().parents[1] / "data_preprocess" / "config.yaml"
        self.config_path = Path(config_path) if config_path else default_cfg
        self.preprocesser = None

    def load_data(self):
        data_loader = DataLoader(self.data_path)
        data = data_loader.load_data()
        return data
    
    def split_data(self, data):
        if self.target_column not in data.columns:
            raise KeyError(f"Target column '{self.target_column}' not found in data.")
        X = data.drop(self.target_column, axis=1)
        y = data[self.target_column]
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

    def preprocess_data(self, config_path=None):
        # allow overriding at call time
        if config_path is not None:
            self.config_path = Path(config_path)

        if self.X_train is None or self.X_test is None:
            raise ValueError("X_train/X_test is None. Call split_data(data) before preprocess_data().")
        if not isinstance(self.X_train, pd.DataFrame) or not isinstance(self.X_test, pd.DataFrame):
            raise TypeError("X_train and X_test must be pandas DataFrames before preprocessing. Ensure you passed a DataFrame to split_data().")
        from typing import cast
        X_train_df = cast(pd.DataFrame, self.X_train)
        X_test_df = cast(pd.DataFrame, self.X_test)

        self.preprocesser = DataPreprocesser(config_path=str(self.config_path))
        cfg_num, cfg_cat = self.preprocesser.get_feature_lists()
        train_cols = set(X_train_df.columns)
        num_present = [c for c in cfg_num if c in train_cols]
        cat_present = [c for c in cfg_cat if c in train_cols]
        missing = [c for c in (cfg_num + cfg_cat) if c not in train_cols]
        if missing:
            print(f"Warning: Skipping {len(missing)} missing configured features: {missing}")
        if not num_present and not cat_present:
            raise ValueError("Configured features are not present in training data.")

        # Coerce numeric columns (e.g., blanks ' ' in TotalCharges) to numeric with NaN
        if num_present:
            X_train_df.loc[:, num_present] = X_train_df.loc[:, num_present].apply(pd.to_numeric, errors="coerce")
            X_test_df.loc[:, num_present] = X_test_df.loc[:, num_present].apply(pd.to_numeric, errors="coerce")

        # Build and apply pipeline
        self.preprocesser.create_pipeline(numeric_features=num_present, categorical_features=cat_present)
        self.X_train = self.preprocesser.fit_transform(X_train_df)
        self.X_test = self.preprocesser.transform(X_test_df)

    def train_model(self):
        if self.X_train is None or self.y_train is None:
            raise ValueError("Training data not prepared. Call split_data() and preprocess_data() first.")
        # Assign to locals to satisfy static/type analysis
        X_train = self.X_train
        y_train = self.y_train
        self.model.fit(X_train, y_train)
    
    def evaluate_model(self):
        if self.X_test is None or self.y_test is None:
            raise ValueError("Test data not prepared. Call split_data() and preprocess_data() first.")
        X_test = self.X_test
        y_test = self.y_test
        y_pred = self.model.predict(X_test)
        print("Confusion Matrix:")
        print(confusion_matrix(y_test, y_pred))
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        return y_pred
    
    def predict(self, X):
        if self.preprocesser is not None:
            X_processed = self.preprocesser.transform(X)
        else:
            X_processed = X
        return self.model.predict(X_processed)
    
    def save_model(self, model_path):
        joblib.dump(self.model, model_path)
        print(f"Model saved to {model_path}")
    
    def load_model(self, model_path):
        self.model = joblib.load(model_path)
        print(f"Model loaded from {model_path}")

    # Optional helpers to persist the preprocessor alongside the model
    def save_preprocessor(self, preprocessor_path):
        if self.preprocesser is None:
            raise ValueError("No preprocessor to save. Run preprocess_data() first.")
        joblib.dump(self.preprocesser, preprocessor_path)
        print(f"Preprocessor saved to {preprocessor_path}")

    def load_preprocessor(self, preprocessor_path):
        self.preprocesser = joblib.load(preprocessor_path)
        print(f"Preprocessor loaded from {preprocessor_path}")


def _default_csv_path() -> Path:
    # repo_root / "Telco_Customer_Churn.csv"
    return Path(__file__).resolve().parents[3] / "Telco_Customer_Churn.csv"


if __name__ == "__main__":
    csv_path = _default_csv_path()
    model = Model(data_path=str(csv_path), target_column="Churn")
    data = model.load_data()
    model.split_data(data)
    model.preprocess_data()
    model.train_model()
    model.evaluate_model()
    model.save_model("customer_churn_model.joblib")
    model.save_preprocessor("data_preprocessor.joblib")

