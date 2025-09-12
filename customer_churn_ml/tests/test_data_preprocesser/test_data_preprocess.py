import pytest
import pandas as pd
import numpy as np
import tempfile
import yaml
import sys
from pathlib import Path
from unittest.mock import patch, mock_open

# Add src to path
sys.path.append(str(Path(__file__).parent.parent.parent / "src"))
from data_preprocess.data_preprocesser import DataPreprocesser


class TestDataPreprocesser:
    
    @pytest.fixture
    def sample_config(self):
        return {
            'features': {
                'numeric': ['age', 'income'],
                'categorical': ['gender', 'category']
            },
            'target': 'target',
            'preprocessing': {
                'numeric': {
                    'imputation_strategy': 'mean',
                    'scaling': True
                },
                'categorical': {
                    'imputation_strategy': 'most_frequent',
                    'encoding': 'onehot',
                    'drop_first': True,
                    'handle_unknown': 'ignore'
                }
            }
        }
    
    @pytest.fixture
    def sample_data(self):
        return pd.DataFrame({
            'age': [25, 30, np.nan, 35, 40],
            'income': [50000, 60000, 55000, np.nan, 80000],
            'gender': ['M', 'F', 'M', 'F', np.nan],
            'category': ['A', 'B', 'A', 'C', 'B'],
            'target': [1, 0, 1, 0, 1]
        })
    
    @pytest.fixture
    def temp_config_file(self, sample_config):
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(sample_config, f)
            return f.name
    
    def test_init_with_config_path(self, temp_config_file):
        preprocessor = DataPreprocesser(config_path=temp_config_file)
        assert preprocessor.config is not None
        assert 'features' in preprocessor.config
    
    def test_init_without_config_path(self):
        with patch('builtins.open', mock_open(read_data=yaml.dump({
            'features': {'numeric': [], 'categorical': []},
            'preprocessing': {'numeric': {}, 'categorical': {}}
        }))):
            preprocessor = DataPreprocesser()
            assert preprocessor.config is not None
    
    def test_get_feature_lists(self, temp_config_file):
        preprocessor = DataPreprocesser(config_path=temp_config_file)
        numeric_features, categorical_features = preprocessor.get_feature_lists()
        
        assert numeric_features == ['age', 'income']
        assert categorical_features == ['gender', 'category']
    
    def test_create_pipeline_with_default_features(self, temp_config_file):
        preprocessor = DataPreprocesser(config_path=temp_config_file)
        pipeline = preprocessor.create_pipeline()
        
        assert pipeline is not None
        assert preprocessor.pipeline is not None
        assert len(pipeline.transformers) == 2
    
    def test_create_pipeline_with_custom_features(self, temp_config_file):
        preprocessor = DataPreprocesser(config_path=temp_config_file)
        custom_numeric = ['age']
        custom_categorical = ['gender']
        
        pipeline = preprocessor.create_pipeline(custom_numeric, custom_categorical)
        
        assert pipeline is not None
        assert pipeline.transformers[0][2] == custom_numeric
        assert pipeline.transformers[1][2] == custom_categorical
    
    def test_fit_transform_without_pipeline(self, temp_config_file, sample_data):
        preprocessor = DataPreprocesser(config_path=temp_config_file)
        
        with pytest.raises(ValueError, match="Pipeline has not been created"):
            preprocessor.fit_transform(sample_data)
    
    def test_transform_without_pipeline(self, temp_config_file, sample_data):
        preprocessor = DataPreprocesser(config_path=temp_config_file)
        
        with pytest.raises(ValueError, match="Pipeline has not been created"):
            preprocessor.transform(sample_data)
    
    def test_get_feature_names_without_pipeline(self, temp_config_file):
        preprocessor = DataPreprocesser(config_path=temp_config_file)
        
        with pytest.raises(ValueError, match="Pipeline has not been created"):
            preprocessor.get_feature_names()
    
    def test_fit_transform_success(self, temp_config_file, sample_data):
        preprocessor = DataPreprocesser(config_path=temp_config_file)
        preprocessor.create_pipeline()
        
        result = preprocessor.fit_transform(sample_data[['age', 'income', 'gender', 'category']])
        
        assert isinstance(result, np.ndarray)
        assert result.shape[0] == len(sample_data)
        assert not np.isnan(result).any()
    
    def test_transform_after_fit(self, temp_config_file, sample_data):
        preprocessor = DataPreprocesser(config_path=temp_config_file)
        preprocessor.create_pipeline()
        
        # Fit on training data
        X_train = sample_data[['age', 'income', 'gender', 'category']]
        preprocessor.fit_transform(X_train)
        
        # Transform test data
        X_test = pd.DataFrame({
            'age': [28, 32],
            'income': [52000, 65000],
            'gender': ['M', 'F'],
            'category': ['A', 'B']
        })
        
        result = preprocessor.transform(X_test)
        
        assert isinstance(result, np.ndarray)
        assert result.shape[0] == len(X_test)
    
    def test_get_feature_names_success(self, temp_config_file, sample_data):
        preprocessor = DataPreprocesser(config_path=temp_config_file)
        preprocessor.create_pipeline()
        preprocessor.fit_transform(sample_data[['age', 'income', 'gender', 'category']])
        
        feature_names = preprocessor.get_feature_names()
        
        assert isinstance(feature_names, list)
        assert len(feature_names) > 0
        assert 'age' in feature_names
        assert 'income' in feature_names
    
    def test_pipeline_handles_missing_values(self, temp_config_file, sample_data):
        preprocessor = DataPreprocesser(config_path=temp_config_file)
        preprocessor.create_pipeline()
        
        # Create data with missing values
        data_with_missing = sample_data[['age', 'income', 'gender', 'category']].copy()
        data_with_missing.loc[0, 'age'] = np.nan
        data_with_missing.loc[1, 'gender'] = np.nan
        
        result = preprocessor.fit_transform(data_with_missing)
        
        # Should not contain any NaN values after preprocessing
        assert not np.isnan(result).any()
    
    def test_pipeline_handles_unknown_categories(self, temp_config_file, sample_data):
        preprocessor = DataPreprocesser(config_path=temp_config_file)
        preprocessor.create_pipeline()
        
        # Fit on training data
        X_train = sample_data[['age', 'income', 'gender', 'category']]
        preprocessor.fit_transform(X_train)
        
        # Test data with unknown category
        X_test = pd.DataFrame({
            'age': [28],
            'income': [52000],
            'gender': ['M'],
            'category': ['UNKNOWN']  # This category wasn't in training data
        })
        
        # Should not raise an error due to handle_unknown='ignore'
        result = preprocessor.transform(X_test)
        assert isinstance(result, np.ndarray)
    
    def test_config_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            DataPreprocesser(config_path='nonexistent.yaml')
    
    def test_invalid_config_structure(self):
        invalid_config = {'invalid': 'structure'}
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(invalid_config, f)
            temp_file = f.name
        
        with pytest.raises(KeyError):
            preprocessor = DataPreprocesser(config_path=temp_file)
            preprocessor.get_feature_lists()
    
    def test_pipeline_consistency(self, temp_config_file, sample_data):
        """Test that the same input produces the same output"""
        preprocessor = DataPreprocesser(config_path=temp_config_file)
        preprocessor.create_pipeline()
        
        X = sample_data[['age', 'income', 'gender', 'category']]
        
        # Fit and transform
        result1 = preprocessor.fit_transform(X)
        
        # Transform the same data again
        result2 = preprocessor.transform(X)
        
        # Results should be identical
        np.testing.assert_array_equal(result1, result2)