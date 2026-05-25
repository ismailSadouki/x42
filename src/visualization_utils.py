import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix, brier_score_loss, roc_curve, auc
from sklearn.calibration import calibration_curve
from sklearn.decomposition import PCA


def plot_feature_importance(model, X_train, importance_type='both', top_n=20):
    importance_df = pd.DataFrame({
        'feature': X_train.columns,
        'gain': model.feature_importance(importance_type='gain'),
        'split': model.feature_importance(importance_type='split')
    })
    
    if importance_type == 'gain':
        df = importance_df.nlargest(top_n, 'gain').sort_values('gain')
        plt.figure(figsize=(10,6))
        plt.barh(df['feature'], df['gain'], color='skyblue')
        plt.xlabel("Gain Importance")
        plt.title(f"Top {top_n} Features by Gain")
        plt.show()
    elif importance_type == 'split':
        df = importance_df.nlargest(top_n, 'split').sort_values('split')
        plt.figure(figsize=(10,6))
        plt.barh(df['feature'], df['split'], color='lightgreen')
        plt.xlabel("Split Importance")
        plt.title(f"Top {top_n} Features by Split")
        plt.show()
    else:  # both
        df_gain = importance_df.nlargest(top_n, 'gain').sort_values('gain')
        df_split = importance_df.nlargest(top_n, 'split').sort_values('split')
        fig, axes = plt.subplots(1, 2, figsize=(16,8))
        axes[0].barh(df_gain['feature'], df_gain['gain'], color='skyblue')
        axes[0].set_title(f"Top {top_n} Features by Gain")
        axes[1].barh(df_split['feature'], df_split['split'], color='lightgreen')
        axes[1].set_title(f"Top {top_n} Features by Split")
        plt.tight_layout()
        plt.show()
        
    return importance_df


def shap_summary(model, X, n_samples=5000, large_data=False):

    if large_data:
        X_sample = X.sample(n=min(n_samples, len(X)), random_state=42)
        explainer = shap.Explainer(model)
        shap_values = explainer(X_sample)
        shap.summary_plot(shap_values, X_sample)
        for i in range(min(5, len(X_sample))):
            shap.plots.waterfall(shap_values[i])
    else: 
        explainer = shap.Explainer(model)
        shap_values = explainer(X)

        shap.plots.waterfall(shap_values[0])
        shap.summary_plot(shap_values, X)


def plot_learning_curve(fold_evals, metric_name='auc', show_std=True):
    # Use the provided metric_name, fallback to first key if not found
    first_fold = fold_evals[0]
    if metric_name not in first_fold['train']:
        metric_name = list(first_fold['train'].keys())[0]

    max_len = max(len(f['train'].get(metric_name, [])) for f in fold_evals)

    def pad_scores(fold_evals, key):
        train_arr = np.array([
            np.pad(f['train'].get(key, []), (0, max_len - len(f['train'].get(key, []))), mode='edge')
            for f in fold_evals
        ])
        valid_arr = np.array([
            np.pad(f['valid'].get(key, []), (0, max_len - len(f['valid'].get(key, []))), mode='edge')
            for f in fold_evals
        ])
        return train_arr, valid_arr

    train, valid = pad_scores(fold_evals, metric_name)
    train_mean, train_std = train.mean(axis=0), train.std(axis=0)
    valid_mean, valid_std = valid.mean(axis=0), valid.std(axis=0)

    plt.figure(figsize=(8,5))
    plt.plot(train_mean, label='Train')
    plt.plot(valid_mean, label='Validation')
    if show_std:
        plt.fill_between(range(train_mean.shape[0]), train_mean-train_std, train_mean+train_std, alpha=0.2)
        plt.fill_between(range(valid_mean.shape[0]), valid_mean-valid_std, valid_mean+valid_std, alpha=0.2)
    plt.xlabel("Iterations")
    plt.ylabel(metric_name.upper())
    plt.title("Learning Curve from CV folds")
    plt.legend()
    plt.grid(True)
    plt.show()

    return train_mean, valid_mean, train_std, valid_std












def classification_error_analysis_v2(
    model,
    X,
    y_true,
    y_pred_prob,
    task="binary",
    threshold=None,
    top_features=None,
    save_path=None,
    max_display_errors=50,
    shap_sample_size=5000,
    imbalance_warning_threshold=1.5
):
    """
    Full Advanced Classification Error Analysis (binary & multiclass) with:
    - Confusion matrix & normalized
    - Class-wise error + imbalance-aware impact
    - Error probability, FP/FN distributions, uncertainty
    - Feature-level error rate
    - PCA + clustering of misclassified samples
    - Calibration curve (binary)
    - Top worst errors CSV + display
    - SHAP explanations (worst errors, FN, per-class)
    - Cumulative gains / lift chart (binary)
    """

    df = X.copy()

    # ---------------------------
    # 0️⃣ Determine predicted classes
    # ---------------------------
    if task=="binary":
        if threshold is None:
            from sklearn.metrics import precision_recall_curve
            precision, recall, thresh = precision_recall_curve(y_true, y_pred_prob)
            f1_scores = 2 * (precision * recall) / (precision + recall + 1e-12)
            best_idx = np.argmax(f1_scores)
            threshold = thresh[best_idx] if best_idx < len(thresh) else 0.5
            print(f"✅ Optimal threshold: {threshold:.3f}")
        y_pred = (y_pred_prob >= threshold).astype(int)
        prob_max = y_pred_prob
    else:
        y_pred = y_pred_prob.argmax(axis=1)
        prob_max = np.max(y_pred_prob, axis=1)

    df["true"] = y_true
    df["pred"] = y_pred
    df["pred_prob"] = prob_max
    df["error"] = y_pred != y_true
    df["uncertainty"] = np.abs(prob_max - 0.5) if task=="binary" else 1 - prob_max

    print(f"\nTotal errors: {df['error'].sum()} / {len(y_true)} ({df['error'].mean()*100:.2f}%)")

    # ---------------------------
    # 1️⃣ Confusion matrix
    # ---------------------------
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title("Confusion Matrix")
    plt.show()

    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    plt.figure(figsize=(5,4))
    sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Reds")
    plt.title("Normalized Confusion Matrix")
    plt.show()

    # ---------------------------
    # 2️⃣ Class-wise error + imbalance awareness
    # ---------------------------
    class_counts = df["true"].value_counts().sort_index()
    class_error = df.groupby("true")["error"].mean()
    error_impact = class_error / (class_counts / len(df))
    summary_list = []

    for cls in np.unique(y_true):
        cls_mask = (y_true == cls)
        cls_df = df.loc[cls_mask]
        total = len(cls_df)
        errors_cls = cls_df["error"].sum()
        fn = ((df.true == cls) & (df.pred != cls)).sum() if task!="binary" else ((df.true == cls) & (df.pred == 0)).sum() if cls==1 else ((df.true == 0) & (df.pred == 1)).sum()
        fp = ((df.true != cls) & (df.pred == cls)).sum() if task!="binary" else ((df.true == 0) & (df.pred == 1)).sum() if cls==1 else ((df.true == 1) & (df.pred == 0)).sum()
        avg_conf_error = cls_df.loc[cls_df.error, "pred_prob"].mean() if errors_cls>0 else 0
        avg_uncertainty = cls_df["uncertainty"].mean()
        summary_list.append({
            "class": cls,
            "support": total,
            "error_rate": errors_cls / total,
            "false_negatives": fn,
            "false_positives": fp,
            "avg_confidence_on_errors": avg_conf_error,
            "avg_uncertainty": avg_uncertainty,
            "imbalance_error_impact": error_impact[cls]
        })

    summary_df = pd.DataFrame(summary_list)
    print("\nClass-wise error summary:")
    display(summary_df)

    high_impact = summary_df[summary_df["imbalance_error_impact"] > imbalance_warning_threshold]
    if not high_impact.empty:
        print("\n⚠️ Classes with disproportionately high error relative to size:")
        display(high_impact)

    # ---------------------------
    # 3️⃣ Quick diagnosis
    # ---------------------------
    for _, row in summary_df.iterrows():
        if row["error_rate"] > 0.2:
            print(f"⚠️ High error rate detected for class {row['class']}")
        if row["avg_confidence_on_errors"] > 0.8:
            print(f"⚠️ Model is overconfident on wrong predictions for class {row['class']}")
        if row["false_negatives"] > row["false_positives"]:
            print(f"⚠️ Model misses too many true class {row['class']} (FN problem → recall issue)")

    # ---------------------------
    # 4️⃣ Error probability, FP/FN, uncertainty
    # ---------------------------
    plt.figure(figsize=(6,4))
    sns.histplot(df.loc[df.error, "pred_prob"], bins=40, color="tomato")
    plt.title("Error Distribution")
    plt.show()

    if task=="binary":
        fp_vals = df[(df.pred==1)&(df.true==0)]["pred_prob"]
        fn_vals = df[(df.pred==0)&(df.true==1)]["pred_prob"]
        plt.figure(figsize=(6,4))
        if len(fp_vals)>0: sns.kdeplot(fp_vals,label="FP",fill=True)
        if len(fn_vals)>0: sns.kdeplot(fn_vals,label="FN",fill=True)
        plt.title("FP vs FN Probability Distribution")
        plt.legend()
        plt.show()

    plt.figure(figsize=(6,4))
    sns.histplot(df["uncertainty"], bins=50, color="purple")
    plt.title("Prediction Uncertainty")
    plt.show()

    # ---------------------------
    # 5️⃣ Feature-level error
    # ---------------------------
    if top_features is not None:
        features = [f for f in top_features if f in df.columns]
        
        n_cols = 4
        n_rows = int(np.ceil(len(features) / n_cols))

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 4 * n_rows))
        axes = axes.flatten()

        for i, feature in enumerate(features):
            grouped = df.groupby(feature)["error"].mean().sort_values(ascending=False).head(20)
            
            grouped.plot(
                kind="bar",
                ax=axes[i],
                color="tomato"
            )
            
            axes[i].set_title(f"Error Rate: {feature}")
            axes[i].set_ylabel("Error rate")

        # turn off empty subplots
        for j in range(i + 1, len(axes)):
            axes[j].axis("off")

        plt.tight_layout()
        plt.show()

    # ---------------------------
    # 6️⃣ Top worst errors
    # ---------------------------
    worst_errors = df.loc[df.error].sort_values(by="pred_prob", ascending=False).head(max_display_errors)
    if save_path:
        worst_errors.to_csv(f"{save_path}/top_worst_errors.csv", index=False)
    print("\nTop worst errors:")
    display(worst_errors.head())

    # ---------------------------
    # 7️⃣ PCA + clustering
    # ---------------------------
    if df[df.error].shape[0] > 2:
        X_errors = df.loc[df.error, X.columns]
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_errors)
        kmeans = KMeans(n_clusters=min(5,len(X_errors)), random_state=42)
        clusters = kmeans.fit_predict(X_pca)
        plt.figure(figsize=(6,4))
        plt.scatter(X_pca[:,0],X_pca[:,1],c=clusters,cmap="tab10",alpha=0.6)
        plt.title("PCA + Clustering of Misclassified Samples")
        plt.xlabel("PCA1")
        plt.ylabel("PCA2")
        plt.show()

    # ---------------------------
    # 8️⃣ Calibration curve
    # ---------------------------
    if task=="binary":
        prob_true, prob_pred = calibration_curve(y_true, prob_max, n_bins=10)
        plt.figure(figsize=(6,5))
        plt.plot(prob_pred, prob_true, marker='o')
        plt.plot([0,1],[0,1],linestyle='--',color='gray')
        plt.title("Calibration Curve")
        plt.show()
    # ---------------------------
    # 9️⃣ SHAP explanations (robust for binary & multiclass)
    # ---------------------------
    if len(worst_errors) > 0:
        explainer = shap.TreeExplainer(model)
        
        # Sample subset for speed
        X_shap_sample = worst_errors[X.columns].sample(
            min(len(worst_errors), shap_sample_size), random_state=42
        )

        shap_values_raw = explainer.shap_values(X_shap_sample)

        # Binary case
        if task == "binary":
            shap_values = shap_values_raw[1] if isinstance(shap_values_raw, list) else shap_values_raw
            shap.summary_plot(shap_values, X_shap_sample, plot_type="bar", show=True)

        # Multiclass case
        else:
            # If shap_values_raw is a list of arrays (one per class)
            if isinstance(shap_values_raw, list):
                for cls_idx, cls_array in enumerate(shap_values_raw):
                    # Take only as many rows as X_shap_sample has
                    min_rows = min(cls_array.shape[0], X_shap_sample.shape[0])
                    cls_shap = cls_array[:min_rows, :X_shap_sample.shape[1]]
                    X_cls = X_shap_sample.iloc[:min_rows, :]

                    print(f"\nClass {cls_idx} SHAP summary:")
                    shap.summary_plot(cls_shap, X_cls, plot_type="bar", show=True)
            else:
                # If shap_values_raw is a single array
                shap.summary_plot(shap_values_raw, X_shap_sample, plot_type="bar", show=True)

    # ---------------------------
    # 🔟 Cumulative gains / lift chart (binary)
    # ---------------------------
    if task=="binary":
        df_sorted = df.sort_values("pred_prob", ascending=False)
        df_sorted["cum_positives"] = df_sorted["true"].cumsum()
        plt.figure(figsize=(6,4))
        plt.plot(np.arange(len(df_sorted))/len(df_sorted),
                 df_sorted["cum_positives"]/df_sorted["true"].sum(),
                 label="Model")
        plt.plot([0,1],[0,1], linestyle='--', color='gray', label="Random")
        plt.title("Cumulative Gains / Lift Chart")
        plt.xlabel("Fraction of Data")
        plt.ylabel("Fraction of Positives Captured")
        plt.legend()
        plt.show()

    print("\n✅ Full advanced classification error analysis completed.")

    return df, summary_df, worst_errors




"""
Advanced Classification Error Analysis
=======================================
Supports: LightGBM (Booster & LGBMClassifier) and CatBoost (CatBoostClassifier)
Tailored for: Binary imbalanced tabular data (76/24 class ratio)
              with anonymised column names and mixed feature types.

Usage
-----
    from classification_error_analysis import classification_error_analysis

    df_errors, summary, worst = classification_error_analysis(
        model        = final_lgbm_model,          # or catboost model
        model_type   = "lgbm",                    # "lgbm" | "catboost"
        X            = X_test,
        y_true       = y_test,
        top_features = list(X_test.columns),
        task         = "binary",
        threshold    = None,                      # auto-optimise for F1
        save_path    = "./error_analysis_output",
    )
"""

from __future__ import annotations

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import shap

from sklearn.decomposition       import PCA
from sklearn.cluster             import KMeans
from sklearn.calibration         import calibration_curve
from sklearn.metrics             import (
    confusion_matrix,
    classification_report,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    average_precision_score,
    f1_score,
    brier_score_loss,
)
from sklearn.preprocessing       import StandardScaler
from scipy.stats                 import ks_2samp, mannwhitneyu

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# Colour palette consistent with the EDA (blue = class 0, orange = class 1)
# ─────────────────────────────────────────────────────────────────────────────
_PALETTE = {
    "class0":   "#4C72B0",   # blue
    "class1":   "#DD8452",   # orange
    "error":    "#C44E52",   # red
    "correct":  "#55A868",   # green
    "fp":       "#DD8452",
    "fn":       "#C44E52",
    "neutral":  "#8172B2",
    "grid":     "#E8E8E8",
}
plt.rcParams.update({
    "figure.dpi":        120,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.grid":         True,
    "grid.color":        _PALETTE["grid"],
    "grid.linewidth":    0.5,
})


# ─────────────────────────────────────────────────────────────────────────────
# 0. Prediction helper — model-agnostic interface
# ─────────────────────────────────────────────────────────────────────────────
def _get_predictions(model, X: pd.DataFrame, model_type: str,
                     task: str) -> np.ndarray:
    """
    Returns probability array.
      - Binary  → 1-D array of class-1 probabilities
      - Multi   → 2-D array (n_samples, n_classes)
    Handles LightGBM native Booster, LGBMClassifier, and CatBoostClassifier.
    """
    mt = model_type.lower()

    # ── LightGBM native Booster ───────────────────────────────────────────
    if mt == "lgbm":
        import lightgbm as lgb
        if isinstance(model, lgb.Booster):
            # native API: predict() returns raw probabilities for binary
            raw = model.predict(X, num_iteration=model.best_iteration)
        else:
            # sklearn wrapper
            if task == "binary":
                raw = model.predict_proba(X)[:, 1]
            else:
                raw = model.predict_proba(X)
        return raw

    # ── CatBoost ─────────────────────────────────────────────────────────
    elif mt == "catboost":
        import catboost as cb
        if task == "binary":
            # predict_proba returns (n, 2) → take class-1 column
            raw = model.predict_proba(X)[:, 1]
        else:
            raw = model.predict_proba(X)
        return raw

    else:
        raise ValueError(f"model_type must be 'lgbm' or 'catboost', got '{model_type}'")


def _get_shap_explainer(model, X_sample: pd.DataFrame, model_type: str):
    """Returns a SHAP TreeExplainer and shap_values for X_sample."""
    mt = model_type.lower()
    explainer = shap.TreeExplainer(model)
    sv = explainer.shap_values(X_sample)
    return explainer, sv


# ─────────────────────────────────────────────────────────────────────────────
# Main function
# ─────────────────────────────────────────────────────────────────────────────
def classification_error_analysis(
    model,
    X:                          pd.DataFrame,
    y_true:                     pd.Series | np.ndarray,
    model_type:                 str   = "lgbm",       # "lgbm" | "catboost"
    task:                       str   = "binary",
    threshold:                  float | None = None,
    top_features:               list | None  = None,
    save_path:                  str  | None  = None,
    max_display_errors:         int   = 50,
    shap_sample_size:           int   = 2000,
    imbalance_warning_threshold:float = 1.5,
    n_error_clusters:           int   = 5,
    verbose:                    bool  = True,
):
    """
    Full Advanced Classification Error Analysis for LightGBM AND CatBoost.

    Parameters
    ----------
    model                : fitted LightGBM Booster / LGBMClassifier
                           OR CatBoostClassifier
    X                    : feature DataFrame (test or train)
    y_true               : true labels (0/1 for binary)
    model_type           : "lgbm" or "catboost"
    task                 : "binary" or "multiclass"
    threshold            : decision threshold; None = auto-optimise F1
    top_features         : list of column names for feature-level error plots
    save_path            : directory to save CSVs and PNGs; None = no saving
    max_display_errors   : rows in "worst errors" table
    shap_sample_size     : max samples for SHAP computation
    imbalance_warning_threshold : flag classes with error-impact ratio > this
    n_error_clusters     : k-means clusters on misclassified samples
    verbose              : print progress messages

    Returns
    -------
    df_full     : DataFrame with predictions, errors, uncertainty appended
    summary_df  : per-class error summary
    worst_errors: DataFrame of worst misclassified samples
    metrics_dict: dict of all scalar metrics (AUC, Brier, etc.)
    """

    # ── Setup ────────────────────────────────────────────────────────────────
    if save_path:
        os.makedirs(save_path, exist_ok=True)

    def _save(fig, name: str):
        if save_path:
            fig.savefig(os.path.join(save_path, name),
                        bbox_inches="tight", dpi=150)
        plt.show()
        plt.close(fig)

    y_true = np.asarray(y_true)
    X      = X.reset_index(drop=True)

    # ── Get probabilities ────────────────────────────────────────────────────
    if verbose:
        print(f"\n{'='*65}")
        print(f"  Advanced Error Analysis  |  model={model_type.upper()}  |  task={task}")
        print(f"{'='*65}")

    y_pred_prob = _get_predictions(model, X, model_type, task)

    # ── 0. Threshold ─────────────────────────────────────────────────────────
    if task == "binary":
        if threshold is None:
            prec, rec, thrs = precision_recall_curve(y_true, y_pred_prob)
            f1s             = 2 * prec * rec / (prec + rec + 1e-12)
            best_idx        = np.argmax(f1s[:-1])
            threshold       = float(thrs[best_idx]) if best_idx < len(thrs) else 0.5
            if verbose:
                print(f"\n✅ Auto-threshold (max F1): {threshold:.4f}")

        y_pred   = (y_pred_prob >= threshold).astype(int)
        prob_max = y_pred_prob

    else:   # multiclass
        y_pred   = y_pred_prob.argmax(axis=1)
        prob_max = y_pred_prob.max(axis=1)

    # ── Build analysis DataFrame ──────────────────────────────────────────────
    df = X.copy()
    df["_true"]        = y_true
    df["_pred"]        = y_pred
    df["_pred_prob"]   = prob_max
    df["_error"]       = (y_pred != y_true).astype(int)
    df["_uncertainty"] = (np.abs(prob_max - 0.5)
                          if task == "binary"
                          else 1 - prob_max)

    if task == "binary":
        df["_error_type"] = np.select(
            [
                (df._pred == 1) & (df._true == 0),
                (df._pred == 0) & (df._true == 1),
                (df._pred == 1) & (df._true == 1),
                (df._pred == 0) & (df._true == 0),
            ],
            ["FP", "FN", "TP", "TN"],
            default="??",
        )

    n_errors = int(df["_error"].sum())
    n_total  = len(df)
    if verbose:
        print(f"\n📊 Total errors: {n_errors:,} / {n_total:,} "
              f"({n_errors/n_total*100:.2f}%)")

    # ═════════════════════════════════════════════════════════════════════════
    # 1. GLOBAL METRICS
    # ═════════════════════════════════════════════════════════════════════════
    if verbose:
        print(f"\n{'─'*65}\n1. Global Metrics\n{'─'*65}")

    metrics_dict = {}

    if task == "binary":
        metrics_dict["roc_auc"]       = roc_auc_score(y_true, prob_max)
        metrics_dict["pr_auc"]        = average_precision_score(y_true, prob_max)
        metrics_dict["brier"]         = brier_score_loss(y_true, prob_max)
        metrics_dict["f1_class0"]     = f1_score(y_true, y_pred, pos_label=0)
        metrics_dict["f1_class1"]     = f1_score(y_true, y_pred, pos_label=1)
        metrics_dict["f1_macro"]      = f1_score(y_true, y_pred, average="macro")
        metrics_dict["threshold_used"]= threshold
        metrics_dict["error_rate"]    = n_errors / n_total

        if verbose:
            print(f"  ROC-AUC        : {metrics_dict['roc_auc']:.4f}")
            print(f"  PR-AUC         : {metrics_dict['pr_auc']:.4f}")
            print(f"  Brier score    : {metrics_dict['brier']:.4f}")
            print(f"  F1 class-0     : {metrics_dict['f1_class0']:.4f}")
            print(f"  F1 class-1     : {metrics_dict['f1_class1']:.4f}")
            print(f"  F1 macro       : {metrics_dict['f1_macro']:.4f}")
            print(f"  Threshold      : {threshold:.4f}")
            print(f"\n{classification_report(y_true, y_pred, digits=4)}")

    # ═════════════════════════════════════════════════════════════════════════
    # 2. CONFUSION MATRIX — raw + normalised
    # ═════════════════════════════════════════════════════════════════════════
    if verbose:
        print(f"\n{'─'*65}\n2. Confusion Matrix\n{'─'*65}")

    cm      = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    fig.suptitle("Confusion Matrices", fontsize=13, fontweight="bold")

    for ax, data, fmt, title, cmap in zip(
        axes,
        [cm, cm_norm],
        ["d", ".3f"],
        ["Raw counts", "Row-normalised (recall per class)"],
        ["Blues", "Oranges"],
    ):
        sns.heatmap(data, annot=True, fmt=fmt, cmap=cmap,
                    linewidths=0.5, ax=ax, cbar=False)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")

    fig.tight_layout()
    _save(fig, "01_confusion_matrix.png")

    # ═════════════════════════════════════════════════════════════════════════
    # 3. CLASS-WISE ERROR SUMMARY + IMBALANCE AWARENESS
    # ═════════════════════════════════════════════════════════════════════════
    if verbose:
        print(f"\n{'─'*65}\n3. Class-wise Error Summary\n{'─'*65}")

    class_counts  = pd.Series(y_true).value_counts().sort_index()
    class_error   = df.groupby("_true")["_error"].mean()
    error_impact  = class_error / (class_counts / len(df))

    rows = []
    for cls in np.unique(y_true):
        mask_cls  = df["_true"] == cls
        cls_df    = df[mask_cls]
        n_cls     = len(cls_df)
        n_err     = int(cls_df["_error"].sum())

        if task == "binary":
            fn = int(((df["_true"] == cls) & (df["_pred"] != cls)).sum())
            fp = int(((df["_true"] != cls) & (df["_pred"] == cls)).sum())
        else:
            fn = int(((df["_true"] == cls) & (df["_pred"] != cls)).sum())
            fp = int(((df["_true"] != cls) & (df["_pred"] == cls)).sum())

        avg_conf_err = (cls_df.loc[cls_df["_error"] == 1, "_pred_prob"].mean()
                        if n_err > 0 else np.nan)
        avg_unc      = cls_df["_uncertainty"].mean()

        rows.append({
            "class":                      cls,
            "support":                    n_cls,
            "class_weight_%":             round(n_cls / len(df) * 100, 2),
            "error_rate":                 round(n_err / n_cls, 4),
            "n_errors":                   n_err,
            "false_negatives (FN)":       fn,
            "false_positives (FP)":       fp,
            "FN/FP ratio":                round(fn / (fp + 1e-9), 2),
            "avg_confidence_on_errors":   round(avg_conf_err, 4) if not np.isnan(avg_conf_err) else "—",
            "avg_uncertainty":            round(avg_unc, 4),
            "imbalance_error_impact":     round(error_impact[cls], 3),
        })

    summary_df = pd.DataFrame(rows)
    if verbose:
        print(summary_df.to_string(index=False))

    # Imbalance warning
    high_impact = summary_df[
        summary_df["imbalance_error_impact"] > imbalance_warning_threshold
    ]
    if not high_impact.empty and verbose:
        print(f"\n⚠️  Classes with disproportionately high error relative to class size:")
        print(high_impact[["class", "support", "error_rate",
                            "imbalance_error_impact"]].to_string(index=False))

    # ── Quick diagnostics ────────────────────────────────────────────────────
    if verbose:
        print("\n🔍 Quick Diagnostics:")
    for _, row in summary_df.iterrows():
        if row["error_rate"] > 0.25 and verbose:
            print(f"  ⚠️  Class {row['class']}: high error rate = {row['error_rate']:.3f}")
        if (isinstance(row["avg_confidence_on_errors"], float) and
                row["avg_confidence_on_errors"] > 0.80 and verbose):
            print(f"  ⚠️  Class {row['class']}: model overconfident on wrong predictions "
                  f"(avg_conf = {row['avg_confidence_on_errors']:.3f})")
        if row["false_negatives (FN)"] > row["false_positives (FP)"] and verbose:
            print(f"  ⚠️  Class {row['class']}: FN > FP — recall problem "
                  f"(FN={row['false_negatives (FN)']}, FP={row['false_positives (FP)']})")

    # ═════════════════════════════════════════════════════════════════════════
    # 4. PROBABILITY DISTRIBUTIONS — errors, FP/FN, uncertainty
    # ═════════════════════════════════════════════════════════════════════════
    if verbose:
        print(f"\n{'─'*65}\n4. Probability & Uncertainty Distributions\n{'─'*65}")

    fig = plt.figure(figsize=(16, 4))
    gs  = gridspec.GridSpec(1, 4, figure=fig, wspace=0.35)

    # 4a — Overall score distribution by true class
    ax0 = fig.add_subplot(gs[0])
    for cls, color, lbl in [(0, _PALETTE["class0"], "Class 0 (standard)"),
                             (1, _PALETTE["class1"], "Class 1 (accelerated)")]:
        vals = df.loc[df["_true"] == cls, "_pred_prob"]
        ax0.hist(vals, bins=40, alpha=0.55, color=color, label=lbl, density=True)
    ax0.axvline(threshold, color="black", linestyle="--",
                linewidth=1.2, label=f"Threshold {threshold:.2f}")
    ax0.set_title("Score distribution\nby true class")
    ax0.set_xlabel("Predicted probability (class 1)")
    ax0.legend(fontsize=7)

    # 4b — Score distribution: correct vs errors
    ax1 = fig.add_subplot(gs[1])
    for flag, color, lbl in [(0, _PALETTE["correct"], "Correct"),
                              (1, _PALETTE["error"],   "Error")]:
        vals = df.loc[df["_error"] == flag, "_pred_prob"]
        ax1.hist(vals, bins=40, alpha=0.55, color=color, label=lbl, density=True)
    ax1.axvline(threshold, color="black", linestyle="--", linewidth=1.2)
    ax1.set_title("Score distribution\ncorrect vs. error")
    ax1.set_xlabel("Predicted probability (class 1)")
    ax1.legend(fontsize=7)

    # 4c — FP vs FN score distribution (binary only)
    ax2 = fig.add_subplot(gs[2])
    if task == "binary":
        fp_vals = df.loc[df["_error_type"] == "FP", "_pred_prob"]
        fn_vals = df.loc[df["_error_type"] == "FN", "_pred_prob"]
        if len(fp_vals) > 5:
            sns.kdeplot(fp_vals, ax=ax2, fill=True,
                        color=_PALETTE["fp"], label=f"FP (n={len(fp_vals)})", alpha=0.5)
        if len(fn_vals) > 5:
            sns.kdeplot(fn_vals, ax=ax2, fill=True,
                        color=_PALETTE["fn"], label=f"FN (n={len(fn_vals)})", alpha=0.5)
        ax2.axvline(threshold, color="black", linestyle="--", linewidth=1.2)
        ax2.set_title("FP vs FN\nprobability density")
        ax2.set_xlabel("Predicted probability")
        ax2.legend(fontsize=7)

        # KS test: FP vs FN
        if len(fp_vals) > 5 and len(fn_vals) > 5:
            ks_stat, ks_p = ks_2samp(fp_vals, fn_vals)
            ax2.text(0.02, 0.92, f"KS={ks_stat:.3f}\np={ks_p:.3f}",
                     transform=ax2.transAxes, fontsize=7,
                     bbox=dict(boxstyle="round", fc="white", alpha=0.7))
    else:
        ax2.set_visible(False)

    # 4d — Uncertainty distribution
    ax3 = fig.add_subplot(gs[3])
    for flag, color, lbl in [(0, _PALETTE["correct"], "Correct"),
                              (1, _PALETTE["error"],   "Error")]:
        vals = df.loc[df["_error"] == flag, "_uncertainty"]
        ax3.hist(vals, bins=40, alpha=0.55, color=color, label=lbl, density=True)
    ax3.set_title("Uncertainty\n|prob − 0.5|")
    ax3.set_xlabel("Uncertainty score")
    ax3.legend(fontsize=7)

    fig.suptitle("Probability & Uncertainty Distributions", fontsize=12, fontweight="bold")
    _save(fig, "02_probability_distributions.png")

    # ═════════════════════════════════════════════════════════════════════════
    # 5. ROC + PRECISION-RECALL + THRESHOLD CURVES
    # ═════════════════════════════════════════════════════════════════════════
    if task == "binary":
        if verbose:
            print(f"\n{'─'*65}\n5. ROC / PR / Threshold Analysis\n{'─'*65}")

        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        fig.suptitle("ROC · Precision-Recall · Threshold Analysis",
                     fontsize=12, fontweight="bold")

        # ROC
        fpr, tpr, roc_thrs = roc_curve(y_true, prob_max)
        youden_j = tpr - fpr
        best_j   = np.argmax(youden_j)
        axes[0].plot(fpr, tpr, color=_PALETTE["class1"],
                     label=f"AUC={metrics_dict['roc_auc']:.4f}")
        axes[0].scatter(fpr[best_j], tpr[best_j], color="red", s=80, zorder=5,
                        label=f"Youden J thr={roc_thrs[best_j]:.3f}")
        axes[0].plot([0,1],[0,1], "k--", alpha=0.4, label="Random")
        axes[0].set_title("ROC Curve")
        axes[0].set_xlabel("False Positive Rate")
        axes[0].set_ylabel("True Positive Rate")
        axes[0].legend(fontsize=8)

        # Precision-Recall
        prec, rec, pr_thrs = precision_recall_curve(y_true, prob_max)
        f1s                = 2 * prec * rec / (prec + rec + 1e-9)
        best_f1_idx        = np.argmax(f1s[:-1])
        baseline_pr        = y_true.mean()
        axes[1].plot(rec, prec, color=_PALETTE["class0"],
                     label=f"PR-AUC={metrics_dict['pr_auc']:.4f}")
        axes[1].axhline(baseline_pr, color="k", linestyle="--",
                        alpha=0.4, label=f"Baseline={baseline_pr:.3f}")
        axes[1].scatter(rec[best_f1_idx], prec[best_f1_idx],
                        color="red", s=80, zorder=5,
                        label=f"Best F1 thr={pr_thrs[best_f1_idx]:.3f}")
        axes[1].set_title("Precision-Recall Curve")
        axes[1].set_xlabel("Recall")
        axes[1].set_ylabel("Precision")
        axes[1].legend(fontsize=8)

        # F1 / Precision / Recall vs threshold
        thr_grid = np.linspace(0.05, 0.95, 100)
        f1_cl0, f1_cl1, prec_cl0, rec_cl0 = [], [], [], []
        for t in thr_grid:
            yp = (prob_max >= t).astype(int)
            r  = classification_report(y_true, yp, output_dict=True, zero_division=0)
            f1_cl0.append(r.get("0", {}).get("f1-score", 0))
            f1_cl1.append(r.get("1", {}).get("f1-score", 0))
            prec_cl0.append(r.get("0", {}).get("precision", 0))
            rec_cl0.append(r.get("0", {}).get("recall", 0))

        axes[2].plot(thr_grid, f1_cl0,   color=_PALETTE["class0"], label="F1 class-0")
        axes[2].plot(thr_grid, f1_cl1,   color=_PALETTE["class1"], label="F1 class-1")
        axes[2].plot(thr_grid, prec_cl0, color=_PALETTE["class0"],
                     linestyle="--", alpha=0.6, label="Prec class-0")
        axes[2].plot(thr_grid, rec_cl0,  color=_PALETTE["class0"],
                     linestyle=":",  alpha=0.6, label="Rec class-0")
        axes[2].axvline(threshold, color="black", linestyle="--",
                        linewidth=1.2, label=f"Used={threshold:.3f}")
        axes[2].set_title("Metrics vs Threshold")
        axes[2].set_xlabel("Threshold")
        axes[2].set_ylabel("Score")
        axes[2].legend(fontsize=7)

        fig.tight_layout()
        _save(fig, "03_roc_pr_threshold.png")

    # ═════════════════════════════════════════════════════════════════════════
    # 6. CALIBRATION CURVE
    # ═════════════════════════════════════════════════════════════════════════
    if task == "binary":
        if verbose:
            print(f"\n{'─'*65}\n6. Calibration\n{'─'*65}")

        prob_t, prob_p = calibration_curve(y_true, prob_max, n_bins=10)
        brier          = metrics_dict["brier"]

        fig, ax = plt.subplots(figsize=(6, 5))
        ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Perfect calibration")
        ax.plot(prob_p, prob_t, "o-", color=_PALETTE["class1"],
                label=f"{model_type.upper()} (Brier={brier:.4f})")

        # Shade over/under-confidence regions
        ax.fill_between([0, 1], [0, 1], [0, 0], alpha=0.05, color="blue",
                        label="Under-confident region")
        ax.fill_between([0, 1], [0, 1], [1, 1], alpha=0.05, color="red",
                        label="Over-confident region")

        # Diagnosis
        mean_diff = float(np.mean(prob_t - prob_p))
        if mean_diff > 0.03:
            diag = "Under-confident (probabilities too low)"
        elif mean_diff < -0.03:
            diag = "Over-confident (probabilities too high)"
        else:
            diag = "Well calibrated"

        ax.set_title(f"Calibration Curve\n{diag}")
        ax.set_xlabel("Mean predicted probability")
        ax.set_ylabel("Fraction of positives")
        ax.legend(fontsize=8)
        if verbose:
            print(f"  Diagnosis: {diag}  (mean diff={mean_diff:+.4f})")
        fig.tight_layout()
        _save(fig, "04_calibration.png")

    # ═════════════════════════════════════════════════════════════════════════
    # 7. CUMULATIVE GAINS + LIFT CHART
    # ═════════════════════════════════════════════════════════════════════════
    if task == "binary":
        if verbose:
            print(f"\n{'─'*65}\n7. Gains & Lift\n{'─'*65}")

        df_sorted          = df.sort_values("_pred_prob", ascending=False).copy()
        df_sorted["_rank"] = np.arange(1, len(df_sorted) + 1)
        total_pos          = df_sorted["_true"].sum()

        df_sorted["_cum_pos"]    = df_sorted["_true"].cumsum()
        df_sorted["_pct_data"]   = df_sorted["_rank"] / len(df_sorted)
        df_sorted["_cum_gain"]   = df_sorted["_cum_pos"] / total_pos
        df_sorted["_lift"]       = (df_sorted["_cum_pos"] / df_sorted["_rank"]) / (total_pos / len(df_sorted))

        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        fig.suptitle("Cumulative Gains & Lift Chart", fontsize=12, fontweight="bold")

        # Gains
        axes[0].plot(df_sorted["_pct_data"], df_sorted["_cum_gain"],
                     color=_PALETTE["class1"], label="Model")
        axes[0].plot([0, 1], [0, 1], "k--", alpha=0.5, label="Random")
        axes[0].fill_between(df_sorted["_pct_data"],
                             df_sorted["_cum_gain"],
                             df_sorted["_pct_data"],
                             alpha=0.1, color=_PALETTE["class1"])
        axes[0].set_title("Cumulative Gains")
        axes[0].set_xlabel("Fraction of data (sorted by score)")
        axes[0].set_ylabel("Fraction of positives captured")
        axes[0].legend(fontsize=9)

        # Capture rates at key deciles
        for pct in [0.1, 0.2, 0.3]:
            idx  = int(pct * len(df_sorted))
            gain = df_sorted["_cum_gain"].iloc[idx]
            axes[0].annotate(f"Top {int(pct*100)}%→{gain:.1%}",
                             xy=(pct, gain),
                             xytext=(pct + 0.05, gain - 0.08),
                             fontsize=7, arrowprops=dict(arrowstyle="->", lw=0.8))

        # Lift
        axes[1].plot(df_sorted["_pct_data"], df_sorted["_lift"],
                     color=_PALETTE["class0"], label="Lift")
        axes[1].axhline(1.0, color="k", linestyle="--", alpha=0.5, label="Baseline (lift=1)")
        axes[1].set_title("Lift Chart")
        axes[1].set_xlabel("Fraction of data (sorted by score)")
        axes[1].set_ylabel("Lift")
        axes[1].legend(fontsize=9)

        fig.tight_layout()
        _save(fig, "05_gains_lift.png")

    # ═════════════════════════════════════════════════════════════════════════
    # 8. FEATURE-LEVEL ERROR RATE
    # ═════════════════════════════════════════════════════════════════════════
    if top_features is not None:
        if verbose:
            print(f"\n{'─'*65}\n8. Feature-level Error Rate\n{'─'*65}")

        features = [f for f in top_features if f in df.columns][:20]
        n_cols   = 4
        n_rows   = int(np.ceil(len(features) / n_cols))

        fig, axes = plt.subplots(n_rows, n_cols,
                                 figsize=(n_cols * 5, n_rows * 3.5))
        axes = np.array(axes).flatten()
        fig.suptitle("Feature-level Error Rate\n(how error rate varies with feature value)",
                     fontsize=12, fontweight="bold")

        for i, feat in enumerate(features):
            ser = df[feat]
            n_unique = ser.nunique()

            # Categorical / low-cardinality: bar chart
            if n_unique <= 20 or ser.dtype == object:
                grp = df.groupby(feat)["_error"].mean().sort_values(ascending=False).head(15)
                grp.plot(kind="bar", ax=axes[i], color=_PALETTE["error"], alpha=0.75)
                axes[i].set_xlabel("")
                axes[i].tick_params(axis="x", rotation=45, labelsize=7)
            else:
                # Continuous: bin into deciles, then show error rate per bin
                df["_bin"] = pd.qcut(df[feat], q=10, duplicates="drop", labels=False)
                grp = df.groupby("_bin")["_error"].mean()
                grp.plot(kind="line", ax=axes[i], color=_PALETTE["error"],
                         marker="o", markersize=4)
                axes[i].set_xlabel("Decile")
                df.drop(columns=["_bin"], inplace=True)

            axes[i].set_title(feat, fontsize=9)
            axes[i].set_ylabel("Error rate")

        for j in range(len(features), len(axes)):
            axes[j].set_visible(False)

        fig.tight_layout()
        _save(fig, "06_feature_error_rates.png")

    # ═════════════════════════════════════════════════════════════════════════
    # 9. TOP WORST ERRORS
    # ═════════════════════════════════════════════════════════════════════════
    if verbose:
        print(f"\n{'─'*65}\n9. Top Worst Errors\n{'─'*65}")

    # Worst FP: high score but actually class 0
    # Worst FN: low score but actually class 1
    meta_cols = ["_true", "_pred", "_pred_prob", "_error", "_error_type", "_uncertainty"]

    worst_fp = (df[df["_error_type"] == "FP"]
                .sort_values("_pred_prob", ascending=False)
                .head(max_display_errors // 2))
    worst_fn = (df[df["_error_type"] == "FN"]
                .sort_values("_pred_prob", ascending=True)
                .head(max_display_errors // 2))
    worst_errors = pd.concat([worst_fp, worst_fn]).sort_values("_uncertainty")

    if save_path:
        worst_errors.to_csv(os.path.join(save_path, "worst_errors.csv"))
        worst_fp.to_csv(os.path.join(save_path, "worst_FP.csv"))
        worst_fn.to_csv(os.path.join(save_path, "worst_FN.csv"))
        if verbose:
            print(f"  Saved worst errors to {save_path}/")

    if verbose:
        print(f"\n  Worst False Positives (predicted class 1, actually class 0):")
        print(worst_fp[meta_cols].head(5).to_string(index=False))
        print(f"\n  Worst False Negatives (predicted class 0, actually class 1):")
        print(worst_fn[meta_cols].head(5).to_string(index=False))

    # ═════════════════════════════════════════════════════════════════════════
    # 10. PCA + K-MEANS CLUSTERING OF MISCLASSIFIED SAMPLES
    # ═════════════════════════════════════════════════════════════════════════
    if verbose:
        print(f"\n{'─'*65}\n10. Error Clustering (PCA + K-Means)\n{'─'*65}")

    error_mask = df["_error"] == 1
    n_err_rows = error_mask.sum()

    if n_err_rows > 10:
        num_cols_for_pca = df.select_dtypes(include=np.number).columns.tolist()
        num_cols_for_pca = [c for c in num_cols_for_pca
                            if not c.startswith("_")]

        X_err_raw = df.loc[error_mask, num_cols_for_pca].fillna(0)
        X_scaled  = StandardScaler().fit_transform(X_err_raw)

        pca_2  = PCA(n_components=2, random_state=42)
        X_pca  = pca_2.fit_transform(X_scaled)
        var_exp = pca_2.explained_variance_ratio_

        k      = min(n_error_clusters, n_err_rows - 1, 8)
        km     = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X_pca)

        # Cluster profiles
        cluster_df = X_err_raw.copy()
        cluster_df["_cluster"]    = labels
        cluster_df["_error_type"] = df.loc[error_mask, "_error_type"].values
        cluster_summary           = cluster_df.groupby("_cluster").agg(
            n        = ("_cluster", "count"),
            FP_pct   = ("_error_type", lambda x: (x == "FP").mean()),
            FN_pct   = ("_error_type", lambda x: (x == "FN").mean()),
        )

        fig, axes = plt.subplots(1, 2, figsize=(13, 5))
        fig.suptitle("PCA + K-Means Clustering of Misclassified Samples",
                     fontsize=12, fontweight="bold")

        # Scatter coloured by cluster
        scatter = axes[0].scatter(
            X_pca[:, 0], X_pca[:, 1],
            c=labels, cmap="tab10", alpha=0.6, s=15
        )
        axes[0].set_xlabel(f"PC1 ({var_exp[0]:.1%} var)")
        axes[0].set_ylabel(f"PC2 ({var_exp[1]:.1%} var)")
        axes[0].set_title("Cluster assignment")
        plt.colorbar(scatter, ax=axes[0], label="Cluster")

        # Scatter coloured by FP/FN
        type_map  = {"FP": _PALETTE["fp"], "FN": _PALETTE["fn"],
                     "TP": _PALETTE["correct"], "TN": _PALETTE["class0"]}
        err_types = df.loc[error_mask, "_error_type"].values
        colors    = [type_map.get(t, "grey") for t in err_types]
        axes[1].scatter(X_pca[:, 0], X_pca[:, 1],
                        c=colors, alpha=0.6, s=15)
        from matplotlib.patches import Patch
        legend_el = [Patch(facecolor=type_map[t], label=t)
                     for t in ["FP", "FN"]]
        axes[1].legend(handles=legend_el, fontsize=9)
        axes[1].set_xlabel(f"PC1 ({var_exp[0]:.1%} var)")
        axes[1].set_ylabel(f"PC2 ({var_exp[1]:.1%} var)")
        axes[1].set_title("FP vs FN in PCA space")

        fig.tight_layout()
        _save(fig, "07_error_clustering.png")

        if verbose:
            print(f"\n  Error cluster summary ({k} clusters):")
            print(cluster_summary.to_string())

    # ═════════════════════════════════════════════════════════════════════════
    # 11. SHAP ON WORST ERRORS (FP + FN separately)
    # ═════════════════════════════════════════════════════════════════════════
    if verbose:
        print(f"\n{'─'*65}\n11. SHAP Analysis on Errors\n{'─'*65}")

    feat_cols = [c for c in X.columns if not c.startswith("_")]

    def _run_shap(subset_df: pd.DataFrame, label: str, color: str):
        """Helper: compute and plot SHAP summary for a subset."""
        if len(subset_df) < 3:
            if verbose:
                print(f"  Skipping SHAP for {label}: too few samples ({len(subset_df)})")
            return

        sample_n  = min(len(subset_df), shap_sample_size)
        X_sample  = subset_df[feat_cols].sample(sample_n, random_state=42)

        try:
            explainer   = shap.TreeExplainer(model)
            sv_raw      = explainer.shap_values(X_sample)

            if task == "binary":
                sv = sv_raw[1] if isinstance(sv_raw, list) else sv_raw
            else:
                sv = sv_raw   # list of arrays

            fig_shap, ax_shap = plt.subplots(figsize=(9, 5))
            shap.summary_plot(
                sv if task == "binary" else sv[1],
                X_sample,
                plot_type="bar",
                max_display=15,
                show=False,
                color=color,
            )
            plt.title(f"SHAP Feature Importance — {label} (n={sample_n})",
                      fontsize=11, fontweight="bold")
            plt.tight_layout()
            _save(fig_shap, f"08_shap_{label.lower().replace(' ', '_')}.png")

        except Exception as e:
            if verbose:
                print(f"  ⚠️  SHAP for {label} failed: {e}")

    # SHAP on all errors
    _run_shap(df[df["_error"] == 1], "All_Errors", _PALETTE["error"])

    if task == "binary":
        # SHAP on FP only
        _run_shap(df[df["_error_type"] == "FP"], "False_Positives", _PALETTE["fp"])
        # SHAP on FN only
        _run_shap(df[df["_error_type"] == "FN"], "False_Negatives", _PALETTE["fn"])

    # ═════════════════════════════════════════════════════════════════════════
    # 12. STATISTICAL TESTS — FP vs FN per feature
    # ═════════════════════════════════════════════════════════════════════════
    if task == "binary" and top_features is not None:
        if verbose:
            print(f"\n{'─'*65}\n12. Statistical Separation: FP vs FN (Mann-Whitney U)\n{'─'*65}")

        fp_df = df[df["_error_type"] == "FP"]
        fn_df = df[df["_error_type"] == "FN"]
        stat_rows = []

        for feat in [f for f in top_features if f in df.columns][:30]:
            if df[feat].dtype not in [np.float64, np.float32, np.int64, np.int32]:
                continue
            fp_v = fp_df[feat].dropna()
            fn_v = fn_df[feat].dropna()
            if len(fp_v) < 5 or len(fn_v) < 5:
                continue
            stat, p = mannwhitneyu(fp_v, fn_v, alternative="two-sided")
            stat_rows.append({
                "feature":     feat,
                "FP_mean":     round(fp_v.mean(), 4),
                "FN_mean":     round(fn_v.mean(), 4),
                "FP_median":   round(fp_v.median(), 4),
                "FN_median":   round(fn_v.median(), 4),
                "mwu_stat":    round(stat, 1),
                "p_value":     round(p, 6),
                "significant": "✅" if p < 0.05 else "—",
            })

        stat_df = pd.DataFrame(stat_rows).sort_values("p_value")
        if verbose and len(stat_df) > 0:
            print(stat_df.to_string(index=False))
        if save_path and len(stat_df) > 0:
            stat_df.to_csv(os.path.join(save_path, "fp_vs_fn_stats.csv"), index=False)

    # ═════════════════════════════════════════════════════════════════════════
    # 13. DATASET-SPECIFIC CHECKS — anonymised data / imbalance warnings
    # ═════════════════════════════════════════════════════════════════════════
    if verbose:
        print(f"\n{'─'*65}\n13. Dataset-Specific Sanity Checks\n{'─'*65}")

    checks = []

    # Check: are errors concentrated in a single predicted-probability bucket?
    if task == "binary":
        near_thresh = df[
            (df["_pred_prob"] > threshold - 0.1) &
            (df["_pred_prob"] < threshold + 0.1)
        ]
        near_err_rate = near_thresh["_error"].mean() if len(near_thresh) > 0 else 0
        checks.append(f"  Near-threshold (±0.1) error rate: {near_err_rate:.3f} "
                      f"(n={len(near_thresh)}) "
                      f"{'⚠️  High uncertainty zone' if near_err_rate > 0.3 else '✅'}")

    # Check: imbalance impact on class 0
    c0_row = summary_df[summary_df["class"] == 0]
    if not c0_row.empty:
        c0_err = float(c0_row["error_rate"].values[0])
        c0_imp = float(c0_row["imbalance_error_impact"].values[0])
        checks.append(f"  Class-0 error rate: {c0_err:.4f}  "
                      f"imbalance impact: {c0_imp:.3f}  "
                      f"{'⚠️  Minority class underserved' if c0_imp > 1.5 else '✅'}")

    # Check: are there features with r>0.99 that may cause importance instability?
    num_cols_check = [c for c in df.select_dtypes(np.number).columns
                      if not c.startswith("_")]
    if len(num_cols_check) > 1:
        corr_m     = df[num_cols_check].corr().abs()
        high_corr  = [(corr_m.columns[i], corr_m.columns[j])
                      for i in range(len(corr_m))
                      for j in range(i+1, len(corr_m))
                      if corr_m.iloc[i, j] > 0.97
                      and not corr_m.columns[i].startswith("_")
                      and not corr_m.columns[j].startswith("_")]
        if high_corr:
            checks.append(f"  ⚠️  {len(high_corr)} feature pair(s) with r>0.97 "
                          f"(SHAP importance may be split)")

    for c in checks:
        print(c)

    # ═════════════════════════════════════════════════════════════════════════
    # 14. FINAL SUMMARY CARD
    # ═════════════════════════════════════════════════════════════════════════
    if verbose:
        print(f"\n{'═'*65}")
        print("  FINAL SUMMARY")
        print(f"{'═'*65}")
        print(f"  Model type      : {model_type.upper()}")
        print(f"  Threshold used  : {threshold:.4f}")
        print(f"  Total samples   : {n_total:,}")
        print(f"  Total errors    : {n_errors:,}  ({n_errors/n_total*100:.2f}%)")
        if task == "binary":
            print(f"  ROC-AUC         : {metrics_dict['roc_auc']:.4f}")
            print(f"  PR-AUC          : {metrics_dict['pr_auc']:.4f}")
            print(f"  Brier score     : {metrics_dict['brier']:.4f}")
            print(f"  F1 macro        : {metrics_dict['f1_macro']:.4f}")
            print(f"  F1 class-0      : {metrics_dict['f1_class0']:.4f}")
            print(f"  F1 class-1      : {metrics_dict['f1_class1']:.4f}")
            n_fp = int((df["_error_type"] == "FP").sum())
            n_fn = int((df["_error_type"] == "FN").sum())
            print(f"  False Positives : {n_fp:,}")
            print(f"  False Negatives : {n_fn:,}")
        print(f"{'═'*65}")
        print("✅ Full advanced classification error analysis completed.\n")

    return df, summary_df, worst_errors, metrics_dict