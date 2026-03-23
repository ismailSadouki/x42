

from sklearn.metrics import roc_auc_score, log_loss, accuracy_score
import pandas as pd

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
