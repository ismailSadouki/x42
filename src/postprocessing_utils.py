import optuna
import numpy as np
import logging

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    log_loss,
    f1_score,
    precision_score,
    recall_score,
    balanced_accuracy_score,
    matthews_corrcoef
)

# -------------------------------
# Silence Optuna logs
# -------------------------------
optuna.logging.set_verbosity(optuna.logging.CRITICAL)
logging.getLogger("optuna").setLevel(logging.CRITICAL)


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
    # Metrics
    # -------------------------------
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

    else:
        raise ValueError(f"Unsupported metric: {kaggle_eval}")


# =========================================================
# Post-processing optimizer
# =========================================================
def optimize_postprocessing(
    y_proba,
    y_true,
    task="multiclass",
    kaggle_eval="accuracy",
    n_trials=300,
    seeds=[42],
    reg_strength=0.01,
):
    best_score = -np.inf
    best_params_overall = None
    best_seed = None

    y_proba = np.array(y_proba)

    if kaggle_eval in ["auc", "logloss"]:
        raise ValueError("Postprocessing is not useful for AUC/logloss (needs probabilities)")

    for seed in seeds:
        np.random.seed(seed)

        sampler = optuna.samplers.TPESampler(seed=seed)
        study = optuna.create_study(direction="maximize", sampler=sampler)

        def objective(trial):
            proba = y_proba.copy()

            mode = trial.suggest_categorical(
                "mode",
                [
                    "none",
                    "temperature",
                    "weights",
                    "thresholds",
                    "weights+temperature",
                    "weights+thresholds",
                    "all",
                ],
            )

            # -------------------------------
            # Temperature scaling
            # -------------------------------
            if "temperature" in mode:
                t = trial.suggest_float("temperature", 0.5, 2.0)
                proba = proba ** (1.0 / t)

            # -------------------------------
            # Class weights
            # -------------------------------
            weights = None
            if "weights" in mode and task == "multiclass":
                weights = np.array([
                    trial.suggest_float(f"w_{i}", 0.5, 3.0, log=True)
                    for i in range(proba.shape[1])
                ])
                weights = weights / np.mean(weights)
                proba = proba * weights[np.newaxis, :]

            # -------------------------------
            # Thresholds
            # -------------------------------
            thresholds = None
            if "thresholds" in mode:
                if task == "binary":
                    thresholds = trial.suggest_float("threshold", 0.05, 0.95)
                else:
                    thresholds = np.array([
                        trial.suggest_float(f"thr_{i}", 0.3, 0.9)
                        for i in range(proba.shape[1])
                    ])
                    proba = proba / thresholds[np.newaxis, :]

            # -------------------------------
            # Normalize (CRUCIAL)
            # -------------------------------
            if task == "multiclass":
                proba = proba / np.sum(proba, axis=1, keepdims=True)

            # -------------------------------
            # Evaluate
            # -------------------------------
            score = evaluate_metric(y_true, proba, task, kaggle_eval)

            # -------------------------------
            # Regularization
            # -------------------------------
            reg = 0
            if weights is not None:
                reg += np.sum((weights - 1.0) ** 2)

            if thresholds is not None and task == "multiclass":
                reg += np.var(thresholds)

            return score - reg_strength * reg

        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

        if study.best_value > best_score:
            best_score = study.best_value
            best_params_overall = study.best_params.copy()
            best_seed = seed

    print("\n🔥 BEST STRATEGY:", best_params_overall)
    print("🔥 BEST SEED:", best_seed)
    print("🔥 BEST SCORE:", best_score)

    return best_params_overall, best_seed, best_score


# =========================================================
# Apply post-processing
# =========================================================
def apply_postprocessing(y_proba, params, task="multiclass"):
    """
    Apply post-processing and return FINAL CLASS PREDICTIONS (1D).
    """

    y_proba = np.array(y_proba)

    # Already labels → just return
    if y_proba.ndim == 1:
        return y_proba

    mode = params.get("mode", "none")

    # -------------------------------
    # Temperature
    # -------------------------------
    if "temperature" in mode:
        t = params.get("temperature", 1.0)
        y_proba = y_proba ** (1.0 / t)

    # -------------------------------
    # Weights
    # -------------------------------
    if "weights" in mode and task == "multiclass":
        weights = np.array([
            params[k] for k in sorted(k for k in params if k.startswith("w_"))
        ])
        weights = weights / np.mean(weights)
        y_proba = y_proba * weights[np.newaxis, :]

    # -------------------------------
    # Thresholds
    # -------------------------------
    if "thresholds" in mode:
        if task == "binary":
            thr = params.get("threshold", 0.5)
            proba = y_proba[:, 1] if y_proba.ndim > 1 else y_proba
            return (proba >= thr).astype(int)
        else:
            thresholds = np.array([
                params[k] for k in sorted(k for k in params if k.startswith("thr_"))
            ])
            y_proba = y_proba / thresholds[np.newaxis, :]

    # -------------------------------
    # Normalize (CRUCIAL)
    # -------------------------------
    if task == "multiclass":
        y_proba = y_proba / np.sum(y_proba, axis=1, keepdims=True)

    # -------------------------------
    # Final predictions
    # -------------------------------
    if task == "binary":
        proba = y_proba[:, 1] if y_proba.ndim > 1 else y_proba
        return (proba >= 0.5).astype(int)

    else:
        return np.argmax(y_proba, axis=1)