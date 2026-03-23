import os
import numpy as np
from src.config import Config 

"""
✅ This handles saving/loading:

- OOF predictions

- Test predictions

"""

class OOFManager:
    def __init__(self, base_dir=Config.EXPERIMENTS_DIR):
        self.base_dir = base_dir
        os.makedirs(self.base_dir, exist_ok=True)

    def save_oof(self, preds, exp_name, dataset="train", filename=None):
        """
        Save predictions for a dataset (train OOF or test set).
        - preds: numpy array or list of predictions
        - dataset: 'train' or 'test'
        - filename: optional custom filename
        """
        exp_dir = os.path.join(self.base_dir, exp_name)
        os.makedirs(exp_dir, exist_ok=True)

        if filename is None:
            filename = f"{dataset}_preds.npy"

        path = os.path.join(exp_dir, filename)
        np.save(path, preds)
        print(f"Saved {dataset} predictions to {path}")
        return path

    def load_oof(self, exp_name, dataset="train", filename=None):
        """
        Load predictions for a dataset (train OOF or test set).
        """
        if filename is None:
            filename = f"{dataset}_preds.npy"
        path = os.path.join(self.base_dir, exp_name, filename)
        return np.load(path)