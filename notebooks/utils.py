# plot_utils.py
from sklearn.model_selection import learning_curve
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve, roc_auc_score

from sklearn.model_selection import learning_curve
import matplotlib.pyplot as plt
import numpy as np
import lightgbm as lgb



from sklearn.metrics import accuracy_score, roc_auc_score, log_loss

def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=-1, train_sizes=np.linspace(.1, 1.0, 5),
                        scoring='accuracy'):
    """
    Plot learning curve for scikit-learn estimators and LightGBM Booster.

    scoring: 'accuracy', 'roc_auc', 'log_loss'
    """
    plt.figure(figsize=(8,6))
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")

    train_scores_mean, train_scores_std = [], []
    test_scores_mean, test_scores_std = [], []

    for train_size in train_sizes:
        train_fold_scores, test_fold_scores = [], []

        for train_idx, test_idx in cv.split(X, y):
            n_train = int(len(train_idx) * train_size)
            train_subset_idx = train_idx[:n_train]

            X_train_subset = X.iloc[train_subset_idx] if hasattr(X, "iloc") else X[train_subset_idx]
            y_train_subset = y.iloc[train_subset_idx] if hasattr(y, "iloc") else y[train_subset_idx]
            X_test_fold = X.iloc[test_idx] if hasattr(X, "iloc") else X[test_idx]
            y_test_fold = y.iloc[test_idx] if hasattr(y, "iloc") else y[test_idx]

            # sklearn estimator
            if hasattr(estimator, "fit"):
                est = estimator.__class__(**estimator.get_params())
                est.fit(X_train_subset, y_train_subset)
                if scoring in ['roc_auc', 'log_loss']:
                    y_train_pred = est.predict_proba(X_train_subset)[:,1]
                    y_test_pred = est.predict_proba(X_test_fold)[:,1]
                else:
                    y_train_pred = est.predict(X_train_subset)
                    y_test_pred = est.predict(X_test_fold)

            # LightGBM Booster
            else:
                lgb_train = lgb.Dataset(X_train_subset, label=y_train_subset)
                booster = lgb.train(
                    params=getattr(estimator, 'params', {}),
                    train_set=lgb_train,
                    num_boost_round=getattr(estimator, 'best_iteration', 100)
                )
                y_train_pred = booster.predict(X_train_subset)
                y_test_pred = booster.predict(X_test_fold)

            # Compute score
            if scoring == "accuracy":
                train_score = accuracy_score(y_train_subset, (y_train_pred >= 0.5).astype(int))
                test_score = accuracy_score(y_test_fold, (y_test_pred >= 0.5).astype(int))
            elif scoring == "roc_auc":
                train_score = roc_auc_score(y_train_subset, y_train_pred)
                test_score = roc_auc_score(y_test_fold, y_test_pred)
            elif scoring == "log_loss":
                train_score = -log_loss(y_train_subset, y_train_pred)
                test_score = -log_loss(y_test_fold, y_test_pred)
            else:
                raise ValueError(f"Unsupported scoring: {scoring}")

            train_fold_scores.append(train_score)
            test_fold_scores.append(test_score)

        train_scores_mean.append(np.mean(train_fold_scores))
        train_scores_std.append(np.std(train_fold_scores))
        test_scores_mean.append(np.mean(test_fold_scores))
        test_scores_std.append(np.std(test_fold_scores))

    # Plot
    plt.grid()
    plt.fill_between(train_sizes,
                     np.array(train_scores_mean) - np.array(train_scores_std),
                     np.array(train_scores_mean) + np.array(train_scores_std),
                     alpha=0.1, color="r")
    plt.fill_between(train_sizes,
                     np.array(test_scores_mean) - np.array(test_scores_std),
                     np.array(test_scores_mean) + np.array(test_scores_std),
                     alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Cross-validation score")
    plt.legend(loc="best")
    plt.show()







def plot_roc(y_true, y_proba, label=None, figsize=(10,6), title='ROC Curve'):
    """
    Plot ROC curve and print ROC-AUC score.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        True binary labels.
    y_proba : array-like of shape (n_samples,)
        Target scores, probability estimates of the positive class.
    label : str, optional
        Label for the ROC curve (default is None).
    figsize : tuple, optional
        Figure size (default is (10,6)).
    title : str, optional
        Plot title (default is 'ROC Curve').

    Returns
    -------
    roc_auc : float
        ROC-AUC score.
    """
    # compute FPR, TPR
    fpr, tpr, _ = roc_curve(y_true, y_proba)

    # plot
    plt.figure(figsize=figsize)
    plt.plot(fpr, tpr, linewidth=2, label=label)
    plt.plot([0, 1], [0, 1], 'r', linewidth=2, linestyle='--')
    plt.axis([0, 1, 0, 1])
    plt.xlabel('False Positive Rate (FPR)', fontsize=14)
    plt.ylabel('True Positive Rate (TPR)', fontsize=14)
    plt.title(title, fontsize=16)
    if label:
        plt.legend(loc='lower right')
    plt.show()

    # ROC-AUC score
    roc_auc = roc_auc_score(y_true, y_proba)
    print(f"ROC-AUC Score: {roc_auc:.5f}")

    return roc_auc