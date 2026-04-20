

from sklearn.metrics import f1_score, mean_squared_error, precision_score, recall_score, roc_auc_score, log_loss, accuracy_score
import pandas as pd
import numpy as np


def evaluate_model(y_true, y_pred_prob, threshold=0.5, task='binary'):
    """
    Evaluate ROC AUC, logloss, accuracy for binary or multiclass classification.
    Returns a DataFrame with metrics.
    """
    if task=='binary':
        y_pred_class = (y_pred_prob >= threshold).astype(int)
        roc_auc = roc_auc_score(y_true, y_pred_prob)
    else:
        y_pred_class = y_pred_prob.argmax(axis=1)
        roc_auc = roc_auc_score(y_true, y_pred_prob, multi_class='ovr')
    
    logloss = log_loss(y_true, y_pred_prob)
    acc = accuracy_score(y_true, y_pred_class)
    
    metrics_df = pd.DataFrame({
        'Metric': ['ROC AUC', 'Log Loss', 'Accuracy'],
        'Value': [roc_auc, logloss, acc]
    })
    
    return metrics_df, y_pred_class



# =========================================================
# Metric handler (ROBUST + EXTENSIBLE)
# =========================================================
def evaluate_metric(y_true, y_input, task, kaggle_eval):
    """
    Smart evaluator:
    - Accepts BOTH probabilities and class labels
    - Automatically detects input type
    """

    y_true = np.ravel(y_true)
    y_input = np.array(y_input)


    # -------------------------------
    # 1. Handle Regression Directly (New)
    # -------------------------------
    if task == "regression":
        if kaggle_eval == "rmse":
            return np.sqrt(mean_squared_error(y_true, y_input))
        elif kaggle_eval == "mae":
            return np.mean(np.abs(y_true - y_input))
        elif kaggle_eval == "mse":
            return mean_squared_error(y_true, y_input)
        else:
            raise ValueError(f"Unsupported regression metric: {kaggle_eval}")

    # -------------------------------
    # Detect if input is labels or probabilities
    # -------------------------------
    is_labels = False

    if y_input.ndim == 1:
        # If values are integers → labels
        if np.issubdtype(y_input.dtype, np.integer):
            is_labels = True
        # If only few unique values → likely labels
        elif len(np.unique(y_input)) < 20:
            is_labels = True

    # -------------------------------
    # Build y_pred / y_proba
    # -------------------------------
    if is_labels:
        y_pred = y_input

        # We can't compute AUC/logloss without probabilities
        y_proba = None

    else:
        # probabilities
        if task == "binary":
            if y_input.ndim > 1:
                y_proba = y_input[:, 1]
            else:
                y_proba = y_input

            y_pred = (y_proba >= 0.5).astype(int)

        else:  # multiclass
            y_proba = y_input
            y_pred = np.argmax(y_proba, axis=1)


    # -------------------------------
# Metrics (Classification)
#     # -------------------------------
    if kaggle_eval == "accuracy":
        return accuracy_score(y_true, y_pred)

    elif kaggle_eval == "f1":
        return f1_score(y_true, y_pred, average="weighted")

    elif kaggle_eval == "precision":
        return precision_score(y_true, y_pred, average="weighted")

    elif kaggle_eval == "recall":
        return recall_score(y_true, y_pred, average="weighted")

    elif kaggle_eval == "auc":
        if y_proba is None:
            raise ValueError("AUC requires probabilities, not class labels")
        if task == "binary":
            return roc_auc_score(y_true, y_proba)
        else:
            return roc_auc_score(y_true, y_proba, multi_class="ovr")

    elif kaggle_eval == "logloss":
        if y_proba is None:
            raise ValueError("Logloss requires probabilities, not class labels")
        return -log_loss(y_true, y_proba)
    elif kaggle_eval == "mpa@3":
        if y_proba is None:
            raise ValueError("MPA@3 requires probabilities, not class labels")
        return mpa_at_3(y_true, y_proba)

    else:
        raise ValueError(f"Unsupported metric: {kaggle_eval}")



# ---------------------
# mpa@3
# ---------------------
def mpa_at_3(y_true, y_proba):
    """
    y_true: array-like of shape (n_samples,)
    y_proba: array-like of shape (n_samples, n_classes), predicted probabilities
    """
    top_k = 3
    y_true = np.array(y_true)
    y_proba = np.array(y_proba)

    # Get top 3 predicted indices
    top_preds = np.argsort(-y_proba, axis=1)[:, :top_k]

    score = 0.0
    for i, true_label in enumerate(y_true):
        if true_label in top_preds[i]:
            # weight by position (earlier is better)
            rank = np.where(top_preds[i] == true_label)[0][0]  # 0-based
            score += 1.0 / (rank + 1)
    
    return score / len(y_true)


def get_top_k_predictions(proba, k=3):
    top_k = np.argsort(proba, axis=1)[:, -k:][:, ::-1]
    return top_k.tolist()


