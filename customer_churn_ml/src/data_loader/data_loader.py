import pandas as pd
import os


class DataLoader:
    def __init__(self, file_path: str):
        self.file_path = file_path
        self.data = None

    def load_data(self) -> pd.DataFrame:
        if not os.path.exists(self.file_path):
            raise FileNotFoundError(f"The file {self.file_path} does not exist.")

        try:
            file_extension = os.path.splitext(self.file_path)[1].lower()
            if file_extension == ".csv":
                self.data = pd.read_csv(self.file_path)
            elif file_extension in [".xls", ".xlsx"]:
                self.data = pd.read_excel(self.file_path)
            elif file_extension == ".parquet":
                self.data = pd.read_parquet(self.file_path)
            else:
                raise ValueError(
                    "Unsupported file format. Please provide a CSV, Excel, or Parquet file."
                )

            print(f"Data loaded successfully from {self.file_path}")
            return self.data
        except pd.errors.EmptyDataError:
            raise ValueError(f"File is empty: {self.file_path}")
        except pd.errors.ParserError:
            raise ValueError(f"Error parsing file: {self.file_path}")
        except ValueError:
            raise
        except Exception as e:
            raise Exception(f"Unexpected error loading data: {str(e)}")
