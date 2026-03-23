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












def classification_error_analysis(
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
        for feature in top_features:
            if feature not in df.columns: continue
            grouped = df.groupby(feature)["error"].mean().sort_values(ascending=False)
            plt.figure(figsize=(8,4))
            grouped.head(20).plot(kind="bar", color="tomato")
            plt.title(f"Error Rate by Feature: {feature}")
            plt.ylabel("Error rate")
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

