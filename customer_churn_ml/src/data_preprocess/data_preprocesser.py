## Data Preprocessing Pipeline
import pandas as pd
import yaml
from pathlib import Path
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import numpy as np


class DataPreprocesser:
    def __init__(self, config_path=None):
        self.pipeline = None
        self.config = self._load_config(config_path)

    def _load_config(self, config_path):
        if config_path is None:
            config_path = Path(__file__).parent / "config.yaml"

        with open(config_path, "r") as file:
            return yaml.safe_load(file)

    def get_feature_lists(self):
        numeric_features = self.config["features"]["numerical"]
        categorical_features = self.config["features"]["categorical"]
        return numeric_features, categorical_features

    def create_pipeline(self, numeric_features=None, categorical_features=None):
        if numeric_features is None or categorical_features is None:
            numeric_features, categorical_features = self.get_feature_lists()

        steps_config = self.config["steps"]

        # Create numerical transformer
        numeric_steps = []
        if "imputer" in steps_config["numerical"]:
            imputer_kwargs = steps_config["numerical"].get("imputer_kwargs", {})
            numeric_steps.append(("imputer", SimpleImputer(**imputer_kwargs)))

        if "scaler" in steps_config["numerical"]:
            scaler_kwargs = steps_config["numerical"].get("scaler_kwargs", {})
            numeric_steps.append(("scaler", StandardScaler(**scaler_kwargs)))

        numeric_transformer = Pipeline(steps=numeric_steps)

        # Create categorical transformer
        categorical_steps = []
        if "imputer" in steps_config["categorical"]:
            imputer_kwargs = steps_config["categorical"].get("imputer_kwargs", {})
            categorical_steps.append(("imputer", SimpleImputer(**imputer_kwargs)))

        if "encoder" in steps_config["categorical"]:
            encoder_kwargs = steps_config["categorical"].get("encoder_kwargs", {})
            categorical_steps.append(("encoder", OneHotEncoder(**encoder_kwargs)))

        categorical_transformer = Pipeline(steps=categorical_steps)

        self.pipeline = ColumnTransformer(
            transformers=[
                ("num", numeric_transformer, numeric_features),
                ("cat", categorical_transformer, categorical_features),
            ]
        )

        return self.pipeline

    def fit_transform(self, X):
        if self.pipeline is None:
            raise ValueError(
                "Pipeline has not been created. Call create_pipeline() first."
            )
        return self.pipeline.fit_transform(X)

    def transform(self, X):
        if self.pipeline is None:
            raise ValueError(
                "Pipeline has not been created. Call create_pipeline() first."
            )
        return self.pipeline.transform(X)

    def get_feature_names(self):
        if self.pipeline is None:
            raise ValueError(
                "Pipeline has not been created. Call create_pipeline() first."
            )

        feature_names = []

        # Get numeric feature names
        num_features = self.pipeline.transformers_[0][2]
        feature_names.extend(num_features)

        # Get categorical feature names after one-hot encoding
        cat_transformer = self.pipeline.transformers_[1][1]
        cat_features = self.pipeline.transformers_[1][2]
        if hasattr(cat_transformer.named_steps["onehot"], "get_feature_names_out"):
            cat_feature_names = cat_transformer.named_steps[
                "onehot"
            ].get_feature_names_out(cat_features)
        else:
            cat_feature_names = cat_transformer.named_steps["onehot"].get_feature_names(
                cat_features
            )

        feature_names.extend(cat_feature_names)

        return feature_names
