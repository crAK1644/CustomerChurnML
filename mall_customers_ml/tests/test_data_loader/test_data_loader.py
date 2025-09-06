import pytest
import pandas as pd
import os
import tempfile
import sys
from pathlib import Path
from unittest.mock import patch, mock_open

# Add the src directory to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))
import data_loader.data_loader as dl

DataLoader = dl.DataLoader


class TestDataLoader:
    def test_init(self):
        """Test DataLoader initialization"""
        file_path = "test.csv"
        loader = DataLoader(file_path)
        assert loader.file_path == file_path
        assert loader.data is None

    def test_load_data_file_not_exists(self):
        """Test loading data when file doesn't exist"""
        loader = DataLoader("nonexistent.csv")
        with pytest.raises(
            FileNotFoundError, match="The file nonexistent.csv does not exist."
        ):
            loader.load_data()

    def test_load_data_csv_success(self):
        """Test successful CSV loading"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as tmp:
            tmp.write("name,age\nJohn,25\nJane,30")
            tmp_path = tmp.name

        try:
            loader = DataLoader(tmp_path)
            result = loader.load_data()

            assert isinstance(result, pd.DataFrame)
            assert len(result) == 2
            assert list(result.columns) == ["name", "age"]
            assert loader.data is not None
        finally:
            os.unlink(tmp_path)

    def test_load_data_excel_success(self):
        """Test successful Excel loading"""
        with tempfile.NamedTemporaryFile(suffix=".xlsx", delete=False) as tmp:
            df = pd.DataFrame({"name": ["John", "Jane"], "age": [25, 30]})
            df.to_excel(tmp.name, index=False)
            tmp_path = tmp.name

        try:
            loader = DataLoader(tmp_path)
            result = loader.load_data()

            assert isinstance(result, pd.DataFrame)
            assert len(result) == 2
            assert loader.data is not None
        finally:
            os.unlink(tmp_path)

    def test_load_data_parquet_success(self):
        """Test successful Parquet loading"""
        with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as tmp:
            df = pd.DataFrame({"name": ["John", "Jane"], "age": [25, 30]})
            df.to_parquet(tmp.name, index=False)
            tmp_path = tmp.name

        try:
            loader = DataLoader(tmp_path)
            result = loader.load_data()

            assert isinstance(result, pd.DataFrame)
            assert len(result) == 2
            assert loader.data is not None
        finally:
            os.unlink(tmp_path)

    def test_load_data_unsupported_format(self):
        """Test loading unsupported file format"""
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as tmp:
            tmp.write(b"test content")
            tmp_path = tmp.name

        try:
            loader = DataLoader(tmp_path)
            with pytest.raises(ValueError, match="Unsupported file format"):
                loader.load_data()
        finally:
            os.unlink(tmp_path)

    def test_load_data_empty_file(self):
        """Test loading empty CSV file"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as tmp:
            tmp.write("")
            tmp_path = tmp.name

        try:
            loader = DataLoader(tmp_path)
            with pytest.raises(ValueError, match=f"File is empty: {tmp_path}"):
                loader.load_data()
        finally:
            os.unlink(tmp_path)

    @patch("pandas.read_csv")
    @patch("os.path.exists")
    def test_load_data_parser_error(self, mock_exists, mock_read_csv):
        """Test handling of pandas ParserError"""
        mock_exists.return_value = True
        mock_read_csv.side_effect = pd.errors.ParserError("Parse error")

        loader = DataLoader("test.csv")
        with pytest.raises(ValueError, match="Error parsing file: test.csv"):
            loader.load_data()

    @patch("pandas.read_csv")
    @patch("os.path.exists")
    def test_load_data_unexpected_error(self, mock_exists, mock_read_csv):
        """Test handling of unexpected errors"""
        mock_exists.return_value = True
        mock_read_csv.side_effect = RuntimeError("Unexpected error")

        loader = DataLoader("test.csv")
        with pytest.raises(
            Exception, match="Unexpected error loading data: Unexpected error"
        ):
            loader.load_data()

    def test_load_data_xlsx_extension(self):
        """Test loading .xlsx file extension"""
        with tempfile.NamedTemporaryFile(suffix=".xlsx", delete=False) as tmp:
            df = pd.DataFrame({"name": ["John", "Jane"], "age": [25, 30]})
            df.to_excel(tmp.name, index=False)
            tmp_path = tmp.name

        try:
            loader = DataLoader(tmp_path)
            result = loader.load_data()

            assert isinstance(result, pd.DataFrame)
            assert len(result) == 2
        finally:
            os.unlink(tmp_path)

    @patch("builtins.print")
    def test_load_data_prints_success_message(self, mock_print):
        """Test that success message is printed"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as tmp:
            tmp.write("name,age\nJohn,25")
            tmp_path = tmp.name

        try:
            loader = DataLoader(tmp_path)
            loader.load_data()
            mock_print.assert_called_with(f"Data loaded successfully from {tmp_path}")
        finally:
            os.unlink(tmp_path)
