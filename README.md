# x42 | ML Competition Pipeline

This repository contains a modular workflow for end-to-end machine learning competitions, from initial data ingestion to advanced ensemble modeling.

---

## 📁 Repository Structure

### 🧠 Core Analysis & Processing
* **`eda.ipynb`** / **`clustering.ipynb`**: Exploratory data analysis and unsupervised structure discovery.

### 🧪 Model Development (`/models`)
* **Tuning**: Specialized **Optuna** notebooks for `XGBoost`, `LGBM`, `CatBoost`, and `HistGB`.
* **AutoML**: Integration of `AutoGluon`, `PyCaret`, and `H2O` for rapid prototyping.
* **Feature Ops**: Dedicated scripts for `feature_engineering.ipynb` and `feature_generation.ipynb`.
* **Ensembling**: Strategy implementations for `ensamble.ipynb` and weighted voting.
* **Utilities**: `utils.py` contains shared helper functions for the pipeline.

### 📝 Natural Language Processing (`/nlp`)
* **Architectures**: Implementations for **BERT**, **LSTMs**, Linear Models, and hybrid models.
* **Embeddings**: Utilization of **GloVe** (50d/100d) for vector representations.
* **Pipeline**: End-to-end `cleaning.ipynb` and `preprocessing.ipynb` for text corpora.

### 🧬 Neural Networks (`/NN`)
* Deep learning specific architectures (TensorFlow/Keras/PyTorch).

---

## 📈 Experiments & Outputs
* **`outputs/experiments/`**: Centralized leaderboard tracking all models.

---

### ⚙️ Core Engine (`/src`)
The backbone of the pipeline, containing modular Python scripts for reproducible experiments:
* **`config.py`**: Centralized hyperparameter, path, and global variable management.
* **`data_loader.py` & `data_splitter.py`**: Handles ingestion and robust cross-validation (CV) folding strategies.
* **`experiment_tracker.py`**: Automated logging of scores, parameters, and model versions.
* **`optuna_utils.py`**: Wrappers for efficient Bayesian optimization across different model types.
* **`oof_manager.py`**: Systematic handling of Out-Of-Fold predictions for multi-stage stacking.
* **`evaluation_utils.py` & `visualization_utils.py`**: Standardized metrics and diagnostic plotting.

---

## 🛠️ Usage
1.  Place raw data in `./train.csv` `./test.csv`.
2.  Run `eda.ipynb` to understand feature distributions.
3.  Execute specific model notebooks in `/models` for hyperparameter optimization.
4.  Check `outputs/experiments_summary.csv` to select the best performing candidates for the final ensemble.
