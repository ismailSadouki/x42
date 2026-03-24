import os
import json
import pickle
import time
import joblib
import matplotlib.pyplot as plt
import pandas as pd
import datetime
import logging
import numpy as np

from src.config import Config
from src.oof_manager import OOFManager

import xgboost as xgb




"""
Experiment Tracker

Handles:
- experiment folder creation
- saving models / params / metrics
- saving OOF predictions
- saving metadata
- optional post-processing
"""


class ExperimentTracker:
    def __init__(self, base_dir=Config.EXPERIMENTS_DIR):
        self.base_dir = base_dir
        os.makedirs(self.base_dir, exist_ok=True)

    # =====================================================
    # Core Saving Utilities
    # =====================================================

    def create_experiment(self, name=None):
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        self.exp_name = f"{timestamp}_{name}" if name else f"exp_{timestamp}"
        self.exp_dir = os.path.join(self.base_dir, self.exp_name)
        os.makedirs(self.exp_dir, exist_ok=True)
        return self.exp_dir

    def save_model(self, model, filename="model.pkl"):
        path = os.path.join(self.exp_dir, filename)
        joblib.dump(model, path)
        print(f"Saved model → {path}")
        return path

    def save_params(self, params, filename="params.json"):
        path = os.path.join(self.exp_dir, filename)
        with open(path, "w") as f:
            json.dump(params, f, indent=4)
        print(f"Saved params → {path}")
        return path

    def save_metrics(self, metrics, filename="metrics.json"):
        path = os.path.join(self.exp_dir, filename)
        with open(path, "w") as f:
            json.dump(metrics, f, indent=4)
        print(f"Saved metrics → {path}")
        return path

    def save_plot(self, fig, filename):
        path = os.path.join(self.exp_dir, filename)
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved plot → {path}")
        return path

    def save_training_time(self, seconds, filename="training_time.txt"):
        path = os.path.join(self.exp_dir, filename)
        with open(path, "w") as f:
            f.write(f"{seconds:.2f} seconds")
        print(f"Saved training time → {path}")
        return path

    def save_metadata(
        self,
        model_name,
        cv_score,
        best_iteration,
        params,
        train_time,
        num_samples,
        num_features,
        filename="metadata.json",
    ):
        meta = {
            "model_name": model_name,
            "cv_score": cv_score,
            "best_iteration": best_iteration,
            "params": params,
            "train_time_seconds": train_time,
            "num_samples": num_samples,
            "num_features": num_features,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }

        path = os.path.join(self.exp_dir, filename)
        with open(path, "w") as f:
            json.dump(meta, f, indent=4)

        print(f"Saved metadata → {path}")
        return path

    # =====================================================
    # Internal Helpers (CLEAN 🔥)
    # =====================================================

    def _predict_model(self, model, X, best_iter=None):
        """Unified prediction for XGBoost / LightGBM"""
        if isinstance(model, xgb.Booster):
            d = xgb.DMatrix(X)
            return model.predict(d, iteration_range=(0, best_iter)) if best_iter else model.predict(d)
        else:
            return model.predict(X, num_iteration=best_iter)

    def _handle_postprocessing(
        self,
        oof_preds,
        y_train,
        postprocessing_params,
        task,
        oof_manager,
    ):
        from src.postprocessing_utils import apply_postprocessing, evaluate_metric

        kaggle_eval = postprocessing_params.get("kaggle_eval", "accuracy")

        # Apply PP
        oof_pred_class_opt = apply_postprocessing(
            oof_preds,
            postprocessing_params["best_params"],
            task=task,
        )

        # Decide evaluation input
        if kaggle_eval in ["accuracy", "f1", "precision", "recall"]:
            eval_input = oof_pred_class_opt
        else:
            eval_input = oof_preds

        score = evaluate_metric(
            y_train,
            eval_input,
            task=task,
            kaggle_eval=kaggle_eval,
        )

        print(f"OOF {kaggle_eval} after post-processing: {score:.4f}")

        # Save PP params
        pp_dir = os.path.join(self.exp_dir, "postprocessing")
        os.makedirs(pp_dir, exist_ok=True)

        pp_file = os.path.join(
            pp_dir,
            f"postprocess_best_seed_{postprocessing_params['best_seed']}.pkl",
        )

        with open(pp_file, "wb") as f:
            pickle.dump(postprocessing_params, f)

        print(f"Saved post-processing params → {pp_file}")

        # Save OOF postprocessed
        oof_manager.save_oof(
            oof_pred_class_opt,
            self.exp_name,
            dataset="oof_postprocessed",
        )

        return score

    # =====================================================
    # Main Runner (CLEAN ORCHESTRATOR 🚀)
    # =====================================================

    def run_experiment(
        self,
        model_name: str,
        final_model,
        X_train,
        y_train,
        X_test,
        best_params: dict,
        best_score: float,
        metrics_df,
        train_time: float,
        oof_preds=None,
        task: str = "binary",
        postprocessing_params: dict = None,
        use_postprocessing: bool = False,
        test_ids=None,
        id_col=None,
        target_col=None,
        sample_submission_path=None,
        submit_proba=False,
        int_to_label=None
    ):
        """
        Full experiment pipeline.
        """

        # ---------------- EXPERIMENT SETUP ----------------
        exp_name = generate_exp_name(model_name, best_score, best_params)
        exp_dir = self.create_experiment(name=exp_name)

        oof_manager = OOFManager()
        best_iter = getattr(final_model, "best_iteration", None)

        # ---------------- SAVE CORE ----------------
        self.save_model(final_model)
        self.save_params(best_params)
        self.save_metrics(metrics_df.set_index("Metric").to_dict()["Value"])
        self.save_training_time(train_time)

        self.save_metadata(
            model_name=model_name,
            cv_score=best_score,
            best_iteration=best_iter,
            params=best_params,
            train_time=train_time,
            num_samples=X_train.shape[0],
            num_features=X_train.shape[1],
        )

        update_experiments_summary(self.base_dir, self.exp_name)

        # ---------------- PREDICTIONS ----------------
        y_train_pred = self._predict_model(final_model, X_train, best_iter)
        y_test_pred = self._predict_model(final_model, X_test, best_iter)

        oof_manager.save_oof(y_train_pred, self.exp_name, dataset="train")
        oof_manager.save_oof(y_test_pred, self.exp_name, dataset="test")

        if oof_preds is not None:
            oof_manager.save_oof(oof_preds, self.exp_name, dataset="oof")

        # ---------------- POSTPROCESSING ----------------
        if use_postprocessing and postprocessing_params:
            self._handle_postprocessing(
                oof_preds,
                y_train,
                postprocessing_params,
                task,
                oof_manager,
            )
            if y_test_pred is not None:
                from src.postprocessing_utils import apply_postprocessing

                y_test_postprocessed = apply_postprocessing(
                    y_test_pred,
                    postprocessing_params["best_params"],
                    task=task
                )
        else:
            y_test_postprocessed = None



        # ---------------- DYNAMIC COLUMN NAMES ----------------
        if sample_submission_path is not None:
            sample_submission = pd.read_csv(sample_submission_path)
            id_col = id_col or sample_submission.columns[0]
            target_col = target_col or sample_submission.columns[1]
            if test_ids is None or (isinstance(test_ids, pd.Series) and test_ids.empty):
                test_ids = sample_submission[id_col]
        else:
            id_col = id_col or "Id"
            target_col = target_col or "Target"


        # Convert 2D probabilities to labels if not submitting probabilities
        if not submit_proba:
            if task == "binary" and y_test_pred.ndim > 1:
                y_test_pred = (y_test_pred[:, 1] >= 0.5).astype(int)
            elif task == "multiclass" and y_test_pred.ndim > 1:
                y_test_pred = np.argmax(y_test_pred, axis=1)

        
        # ---------------- KAGGLE SUBMISSIONS ----------------
        if test_ids is not None and y_test_pred is not None:
            # Raw predictions
            self.save_submission(
                test_ids=test_ids,
                y_test_pred=y_test_pred,
                int_to_label=int_to_label,
                exp_name=self.exp_name,
                id_col=id_col,
                target_col=target_col,
                filename_prefix="submission_raw"
            )
            # Post-processed predictions
            if y_test_postprocessed is not None:
                self.save_submission(
                    test_ids=test_ids,
                    y_test_pred=y_test_postprocessed,
                    int_to_label=int_to_label,
                    exp_name=self.exp_name,
                    id_col=id_col,
                    target_col=target_col,
                    filename_prefix="submission_postprocessed"
                )


        print(f"\n✅ Experiment completed → {exp_dir}\n")
        return exp_dir
    
    def save_submission(
        self,
        test_ids,
        y_test_pred,
        int_to_label=None,  # <-- new param
        exp_name=None,
        id_col="Id",
        target_col="Target",
        filename_prefix="submission"
    ):
        """
        Saves Kaggle submission CSV.
        - test_ids: the ID column for submission
        - y_test_pred: predictions (raw or post-processed)
        """
        if test_ids is None or y_test_pred is None:
            print("No predictions or test IDs provided. Skipping submission.")
            return None
        
        # Convert numeric predictions to original string labels if mapping exists
        if int_to_label is not None:
            y_test_pred = [int_to_label[i] for i in y_test_pred]

        submission = pd.DataFrame({
            id_col: test_ids,
            target_col: y_test_pred
        })

        exp_str = exp_name if exp_name else self.exp_name
        filename = f"{filename_prefix}_{exp_str}.csv"
        path = os.path.join(self.exp_dir, filename)
        submission.to_csv(path, index=False)
        print(f"Saved Kaggle submission → {path}")

        return path


# =====================================================
# Utilities
# =====================================================

def generate_exp_name(model_name, best_score, params):
    ts = datetime.datetime.now().strftime("%Y%m%d-%H%M")
    score_str = f"{best_score:.4f}"
    key_params = "_".join(
        [f"{k}{v}" for k, v in params.items() if k in ["learning_rate", "max_depth"]]
    )
    return f"{model_name}_CVScore{score_str}_exp_{ts}_{key_params}"


def update_experiments_summary(base_dir, exp_name, metadata_file="metadata.json"):
    summary_path = os.path.join(base_dir, "experiments_summary.csv")

    with open(os.path.join(base_dir, exp_name, metadata_file), "r") as f:
        meta = json.load(f)

    row = {
        "exp_name": exp_name,
        "model_name": meta["model_name"],
        "cv_score": meta["cv_score"],
        "best_iteration": meta["best_iteration"],
        "train_time_sec": meta["train_time_seconds"],
        "num_samples": meta["num_samples"],
        "num_features": meta["num_features"],
        "timestamp": meta["timestamp"],
    }

    if os.path.exists(summary_path):
        df = pd.read_csv(summary_path)
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    else:
        df = pd.DataFrame([row])

    df.to_csv(summary_path, index=False)
    print(f"Updated experiments summary → {summary_path}")