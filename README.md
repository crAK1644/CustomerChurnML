# CustomerChurnML

Predict customer churn on the Telco Customer Churn dataset with a clean, configurable pipeline: data loading, preprocessing (imputation, encoding, scaling), model training, and evaluation. The project is structured as a small Python package with tests and a runnable script entry point.

## Project structure

```
CustomerChurnML/
├─ Telco_Customer_Churn.csv              # Dataset (CSV)
├─ customer_churn_ml/
│  ├─ pyproject.toml                     # Project dependencies
│  ├─ src/
│  │  ├─ data_loader/
│  │  │  └─ data_loader.py               # Robust CSV/Excel/Parquet loader
│  │  ├─ data_preprocess/
│  │  │  ├─ config.yaml                  # Preprocessing configuration
│  │  │  └─ data_preprocesser.py         # Pipeline: impute/scale/encode
│  │  └─ model/
│  │     └─ model.py                     # Model class + __main__ runner
│  └─ tests/
│     ├─ test_data_loader/
│     └─ test_data_preprocesser/
└─ README.md
```

## Requirements

- Python 3.13 (configured in `pyproject.toml`)
- macOS or Linux (Windows should work with minor path adjustments)

Core dependencies (installed automatically):
- pandas, numpy, scikit-learn, PyYAML, joblib

## Setup

You can install deps with pip or uv. Pick one of the options below.

### Option A: pip (system Python)

```bash
/usr/local/bin/python3 -m pip install -r <(python3 - <<'PY'
import tomllib, sys, pathlib
p = pathlib.Path('customer_churn_ml/pyproject.toml')
data = tomllib.loads(p.read_text())
for dep in data['project']['dependencies']:
	print(dep)
PY
)
```

### Option B: uv (fast Python package manager)

If you have `uv` installed:

```bash
cd customer_churn_ml
uv sync
```

## Configuration

Preprocessing and some data settings are controlled via `customer_churn_ml/src/data_preprocess/config.yaml`.

Key sections:
- `features.numerical` / `features.categorical`: the feature lists used in the pipeline.
- `steps.numerical` / `steps.categorical`: which imputers/encoders/scalers to use and their kwargs.
- `data.target_column`: default target name (the Model also accepts `target_column`).

Note: The code automatically intersects configured features with the columns present in your data and will warn about any missing ones.

## Quickstart

Run the end-to-end training and evaluation script (uses the CSV shipped at repo root):

```bash
/usr/local/bin/python3 customer_churn_ml/src/model/model.py
```

You should see a confusion matrix and classification report printed at the end.

## Programmatic usage

```python
from customer_churn_ml.src.model.model import Model

model = Model(data_path="Telco_Customer_Churn.csv", target_column="Churn")
data = model.load_data()
model.split_data(data)
model.preprocess_data()              # uses config.yaml by default
model.train_model()
metrics = model.evaluate_model()

# Predict on new raw data (same columns as training features):
# preds = model.predict(new_dataframe)

# Save/load artifacts
model.save_model("models/rf.joblib")
# model.load_model("models/rf.joblib")
```

## Saving the preprocessor (optional)

If you plan to serve predictions on raw feature DataFrames later, persist the preprocessor too:

```python
model.save_preprocessor("models/preprocessor.joblib")
# ... later ...
# model.load_preprocessor("models/preprocessor.joblib")
```

## Testing

Tests live under `customer_churn_ml/tests`. You can run them with `pytest`:

```bash
cd customer_churn_ml
pytest -q
```

## Troubleshooting

- ModuleNotFoundError when running the script: the script includes a fallback to import modules when run directly; ensure you’re invoking the file from the repo root as shown in Quickstart.
- Missing configured features: the script prints which configured features are absent in your CSV; either add them to the data or remove them from `config.yaml`.
- ValueError: Cannot use median strategy with non-numeric data: the pipeline coerces numeric columns to numeric; ensure numerical columns in the CSV aren’t strings with unexpected values.

## License

This project is licensed under the terms of the MIT License. See `LICENSE` for details.