import pandas as pd
from pathlib import Path
from src.config import Config as cfg





def load_data(version="encoded", data_dir=None):
    DATASETS = {

        "raw": {
            "X_train": "X_train.csv",
            "X_test": "X_test.csv",
        },

        "encoded": {
            "X_train": "X_train_encoded.csv",
            "X_test": "X_test_encoded.csv",
        },

        "fe": {
            "X_train": "X_train_fe.csv",
            "X_test": "X_test_fe.csv",
        }

    }

    if version not in DATASETS:
        raise ValueError(
            f"Unknown dataset version '{version}'. "
            f"Available versions: {list(DATASETS.keys())}"
        )
    
    if data_dir is None:
        data_dir = cfg.DATA_DIR
    data_dir = Path(data_dir)


    train_file = DATASETS[version]["X_train"]
    test_file  = DATASETS[version]["X_test"]

    X_train = pd.read_csv(data_dir / train_file)
    X_test  = pd.read_csv(data_dir / test_file)

    y_train = pd.read_csv(data_dir / "y_train.csv")

    # check if y_test exists
    y_test_path = data_dir / "y_test.csv"

    if y_test_path.exists():
        y_test = pd.read_csv(y_test_path)
    else:
        y_test = None

    return X_train, X_test, y_train, y_test


def prepare_data(X_train, X_test, y_train=None, y_test=None, target=None,
                 drop_id=True, verbose=True, label_map=None):
    """
    Flexible data preparation: drops ID column, extracts target, works with
    binary/multiclass labels (numeric or string), encodes labels if necessary,
    or uses a custom mapping.

    Parameters
    ----------
    label_map : dict, optional
        Mapping of label values to integers, e.g. {"A": 1, "B": 0, "C": 2}.
        If provided, this mapping is applied instead of LabelEncoder.

    Returns
    -------
    X_train, X_test, y_train_out, y_test_out, test_ids, num_classes
    """

    # ------------------------
    # Save test IDs
    # ------------------------
    test_ids = X_test[cfg.ID] if cfg.ID in X_test.columns else None

    # ------------------------
    # Drop ID column if requested
    # ------------------------
    if drop_id:
        X_train = X_train.drop(columns=[cfg.ID], errors="ignore")
        X_test  = X_test.drop(columns=[cfg.ID], errors="ignore")

    # ------------------------
    # Handle y_train
    # ------------------------
    if y_train is None:
        y_train_out = None
        y_test_out = None
        num_classes = None
    else:
        if target is not None:
            if isinstance(y_train, pd.DataFrame):
                y_train_out = y_train[target]
            else:
                raise ValueError("y_train must be a DataFrame if target is specified")
            if y_test is not None:
                if isinstance(y_test, pd.DataFrame):
                    y_test_out = y_test[target]
                else:
                    raise ValueError("y_test must be a DataFrame if target is specified")
            else:
                y_test_out = None
        else:
            y_train_out = y_train
            y_test_out = y_test

        # ------------------------
        # Apply label mapping if provided
        # ------------------------
        if label_map is not None:
            y_train_out = y_train_out.map(label_map).astype(int)
            if y_test_out is not None:
                y_test_out = y_test_out.map(label_map).astype(int)
            int_to_label = {v: k for k, v in label_map.items()}
        else:
            # Otherwise, automatically encode strings
            if y_train_out.dtype == "object" or isinstance(y_train_out.iloc[0], str):
                from sklearn.preprocessing import LabelEncoder
                le = LabelEncoder()
                y_train_out = le.fit_transform(y_train_out)
                if y_test_out is not None:
                    y_test_out = le.transform(y_test_out)
                int_to_label = {i: label for i, label in enumerate(le.classes_)}
            else:
                y_train_out = np.array(y_train_out)
                if y_test_out is not None:
                    y_test_out = np.array(y_test_out)
                int_to_label = None

        num_classes = len(np.unique(y_train_out))

    # ------------------------
    # Verbose output
    # ------------------------
    if verbose:
        print(f"Number of classes: {num_classes}")
        print("X_train shape:", X_train.shape)
        print("X_test shape:", X_test.shape)
        if y_train_out is not None:
            print("y_train shape:", y_train_out.shape)
        if y_test_out is not None:
            print("y_test shape:", y_test_out.shape)
        else:
            print("y_test labels are not available")
        if test_ids is not None:
            print("Test IDs available:", len(test_ids))

    return X_train, X_test, y_train_out, y_test_out, test_ids, num_classes, int_to_label





import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split

class AdvancedSplitter:
    """
    Advanced data splitter for ML pipelines.
    
    Features
    --------
    - K-Fold splitting (regular or stratified)
    - Simple train/test split
    - Multiple random states
    - Returns both DataFrames and indices
    """
    
    def __init__(self, test_size=0.2, kfold=True, n_splits=5, stratify=False):
        """
        Parameters
        ----------
        test_size : float, default=0.2
            Fraction of data to use as validation if not using K-Fold.
        kfold : bool, default=True
            Whether to use K-Fold splitting.
        n_splits : int, default=5
            Number of folds for K-Fold.
        stratify : bool, default=False
            If True, uses stratified splitting to preserve class distribution.
        """
        self.test_size = test_size
        self.kfold = kfold
        self.n_splits = n_splits
        self.stratify = stratify

    def split_data(self, X, y, random_state_list=[42]):
        """
        Generator yielding train/validation splits.

        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix.
        y : pd.Series
            Target vector.
        random_state_list : list of int
            Random seeds for reproducibility and multiple splits.

        Yields
        ------
        X_train, X_val : pd.DataFrame
            Train and validation features.
        y_train, y_val : pd.Series
            Train and validation targets.
        train_idx, val_idx : np.ndarray
            Row indices of train/validation splits.
        """
        X = X.reset_index(drop=True)
        y = y.reset_index(drop=True)
        
        if self.kfold:
            for random_state in random_state_list:
                if self.stratify:
                    kf = StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=random_state)
                else:
                    kf = KFold(n_splits=self.n_splits, shuffle=True, random_state=random_state)
                
                for train_idx, val_idx in kf.split(X, y if self.stratify else None):
                    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
                    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
                    yield X_train, X_val, y_train, y_val, train_idx, val_idx
        else:
            stratify_y = y if self.stratify else None
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=self.test_size, random_state=random_state_list[0], stratify=stratify_y
            )
            train_idx = X_train.index.to_numpy()
            val_idx = X_val.index.to_numpy()
            yield X_train, X_val, y_train, y_val, train_idx, val_idx