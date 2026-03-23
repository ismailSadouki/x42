import numpy as np
import os
from pathlib import Path

from sklearn.model_selection import (
    KFold,
    StratifiedKFold,
    GroupKFold,
    RepeatedStratifiedKFold,
    train_test_split
)






class DataSplitter:
    """
    Smart data splitter for ML pipelines with multi-seed CV, fold persistence,
    fold column generation, and verbose info.
    """

    VALID_METHODS = {"kfold", "stratified_kfold", "group_kfold", "repeated_stratified", "train_test"}

    def __init__(
        self,
        method="stratified_kfold",
        n_splits=5,
        n_repeats=1,
        test_size=0.2,
        shuffle=True,
        random_state=42,
        random_states=None,
        folds_path=None
    ):
        if method not in self.VALID_METHODS:
            raise ValueError(f"Invalid split method: {method}")

        self.method = method
        self.n_splits = n_splits
        self.n_repeats = n_repeats
        self.test_size = test_size
        self.shuffle = shuffle
        self.random_state = random_state
        self.random_states = random_states or [random_state]
        self.folds_path = Path(folds_path) if folds_path else None

    # ---------------- Internal splitter ----------------
    def _get_splitter(self, seed):
        if self.method == "kfold":
            return KFold(n_splits=self.n_splits, shuffle=self.shuffle, random_state=seed)
        elif self.method == "stratified_kfold":
            return StratifiedKFold(n_splits=self.n_splits, shuffle=self.shuffle, random_state=seed)
        elif self.method == "group_kfold":
            return GroupKFold(n_splits=self.n_splits)
        elif self.method == "repeated_stratified":
            return RepeatedStratifiedKFold(n_splits=self.n_splits, n_repeats=self.n_repeats, random_state=seed)
        return None

    # ---------------- Create folds ----------------
    def create_folds(self, X, y=None, groups=None):
        if "stratified" in self.method and y is None:
            raise ValueError(f"{self.method} requires y")
        if self.method == "group_kfold" and groups is None:
            raise ValueError("group_kfold requires groups")

        folds = []

        if self.method == "train_test":
            stratify = y if y is not None else None
            train_idx, val_idx = train_test_split(
                np.arange(len(X)),
                test_size=self.test_size,
                random_state=self.random_state,
                stratify=stratify
            )
            folds.append((train_idx, val_idx))
            return folds

        for seed in self.random_states:
            splitter = self._get_splitter(seed)
            if self.method == "group_kfold":
                iterator = splitter.split(X, y, groups)
            elif "stratified" in self.method:
                iterator = splitter.split(X, y)
            else:
                iterator = splitter.split(X)
            for train_idx, val_idx in iterator:
                folds.append((train_idx, val_idx))

        return folds

    # ---------------- Save folds ----------------
    def save_folds(self, folds):
        if self.folds_path is None:
            raise ValueError("folds_path not specified")
        folds_path = Path(self.folds_path)
        folds_path.parent.mkdir(parents=True, exist_ok=True)
        folds = np.array(folds, dtype=object)
        tmp_path = folds_path.parent / (folds_path.stem + ".tmp.npy")
        np.save(tmp_path, folds)
        os.replace(tmp_path, folds_path)

    # ---------------- Load folds ----------------
    def load_folds(self, X=None, y=None, groups=None):
        try:
            if not self.folds_path.exists() or self.folds_path.stat().st_size == 0:
                raise FileNotFoundError("Fold file missing or empty")
            folds = np.load(self.folds_path, allow_pickle=True)
            print(f"✅ Loaded {len(folds)} folds from {self.folds_path}")
            return [(train, val) for train, val in folds]
        except Exception:
            print("⚠️ Fold file missing or corrupted. Regenerating...")
            folds = self.create_folds(X, y, groups)
            if self.folds_path:
                self.save_folds(folds)
            print(f"✅ Created and saved {len(folds)} new folds")
            return folds

    # ---------------- Safe split ----------------
    def safe_split(self, X, y=None, groups=None, reuse_folds=True):
        if reuse_folds and self.folds_path and self.folds_path.exists():
            try:
                folds = self.load_folds(X, y, groups)
                if max([max(val) for _, val in folds]) >= len(X):
                    raise ValueError("Old folds do not match new data")
                print(f"♻️ Reusing existing folds")
                return folds
            except Exception:
                print("⚠️ Regenerating folds for new dataset...")

        folds = self.create_folds(X, y, groups)
        if self.folds_path:
            self.save_folds(folds)
        print(f"✅ Generated {len(folds)} folds for current dataset")
        return folds

    # ---------------- Split ----------------
    def split(self, X, y=None, groups=None, reuse_folds=True, verbose=True):
        folds = self.safe_split(X, y, groups, reuse_folds=reuse_folds)
        if verbose:
            print(f"--- Splitting data ---")
            print(f"Method: {self.method}")
            print(f"Number of splits: {self.n_splits}")
            print(f"Random seeds: {self.random_states}")
            print(f"Dataset size: {len(X)}")
            print(f"Total folds: {len(folds)}\n")

        for i, (train_idx, val_idx) in enumerate(folds):
            if verbose:
                print(f"Fold {i}: Train size={len(train_idx)}, Val size={len(val_idx)}")
            yield train_idx, val_idx

    # ---------------- Number of folds ----------------
    def get_n_folds(self):
        if self.method == "repeated_stratified":
            return self.n_splits * self.n_repeats * len(self.random_states)
        if self.method == "train_test":
            return 1
        return self.n_splits * len(self.random_states)

    # ---------------- Fold column for stacking ----------------
    def create_fold_column(self, df, y=None, groups=None, column_name="fold"):
        df = df.copy()
        df[column_name] = -1
        for fold, (_, val_idx) in enumerate(self.split(df, y, groups)):
            df.loc[val_idx, column_name] = fold
        print(f"✅ Fold column '{column_name}' created in DataFrame")
        return df

#     splitter = DataSplitter(
#     method="stratified_kfold",
#     n_splits=5,
#     random_states=[42,1337,2024]
# )
# Produces:
# 5 folds × 3 seeds = 15 folds

# what this?: Fold column generation
# df = splitter.create_fold_column(train_df, y=train_df.target)