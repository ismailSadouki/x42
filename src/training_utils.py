import lightgbm as lgb


def train_lgb_fold(X_tr, y_tr, X_val, y_val, params, trial=None):

    lgb_train = lgb.Dataset(X_tr, label=y_tr)
    lgb_valid = lgb.Dataset(X_val, label=y_val)

    model = lgb.train(
        params,
        lgb_train,
        num_boost_round=Config.NUM_BOOST_ROUND,
        valid_sets=[lgb_valid],
        callbacks=[
            lgb.early_stopping(Config.EARLY_STOPPING)
        ]
    )

    preds = model.predict(X_val)
    score = roc_auc_score(y_val, preds)

    return model, score











def train_lgbm_final(X_train, y_train, best_params, best_iteration=None, X_val=None, y_val=None):
    """
    Train a final LightGBM model with optional validation set.
    """
    train_set = lgb.Dataset(X_train, label=y_train)
    valid_sets = [train_set] if X_val is None else [train_set, lgb.Dataset(X_val, label=y_val)]
    
    model = lgb.train(
        params=best_params,
        train_set=train_set,
        num_boost_round=best_iteration or best_params.get('num_boost_round', 1000),
        valid_sets=valid_sets,
        callbacks=[
            lgb.log_evaluation(period=100)
            # use this when you have a separate validation set:
            # lgb.early_stopping(stopping_rounds=100, verbose=True),
        ]
    )
    
    return model



import xgboost as xgb

def train_xgb_final(X_train, y_train, best_params, best_iteration=None, X_val=None, y_val=None):
    """
    Train a final XGBoost model with optional validation set.
    """
    # Create DMatrix for training
    dtrain = xgb.DMatrix(X_train, label=y_train)
    
    # Prepare evaluation sets
    evals = [(dtrain, 'train')]
    if X_val is not None and y_val is not None:
        dval = xgb.DMatrix(X_val, label=y_val)
        evals.append((dval, 'valid'))
    
    # Determine number of boosting rounds
    num_boost_round = best_iteration or best_params.get('num_boost_round', 1000)
    
    
    # Train the model
    model = xgb.train(
        params=best_params,
        dtrain=dtrain,
        num_boost_round=num_boost_round,
        evals=evals,
        verbose_eval=100,  # log every 100 iterations
        early_stopping_rounds=100 if X_val is not None else None
    )
    
    return model














# import xgboost as xgb
# import numpy as np
# from sklearn.utils.class_weight import compute_class_weight

# def train_xgb_final(X_train, y_train, best_params, best_iteration=None, X_val=None, y_val=None, task="binary"):
#     """
#     Train a final XGBoost model with optional validation set and class weights.
    
#     Parameters
#     ----------
#     X_train : pd.DataFrame or np.ndarray
#         Training features
#     y_train : pd.Series or np.ndarray
#         Training labels
#     best_params : dict
#         Best hyperparameters from tuning
#     best_iteration : int, optional
#         Number of boosting rounds to train
#     X_val : pd.DataFrame or np.ndarray, optional
#         Validation features
#     y_val : pd.Series or np.ndarray, optional
#         Validation labels
#     task : str
#         "binary" or "multiclass"
#     """
    
#     # -------------------------------
#     # Compute class weights if needed
#     # -------------------------------
#     if task.lower() == "multiclass":
#         classes = np.unique(y_train)
#         class_weights = compute_class_weight("balanced", classes=classes, y=y_train)
#         sample_weights = np.array([class_weights[int(c)] for c in y_train])
#         num_class = len(classes)
#         best_params = best_params.copy()
#         best_params['num_class'] = num_class
#         best_params['objective'] = 'multi:softprob'
#     elif task.lower() == "binary":
#         sample_weights = None
#         best_params = best_params.copy()
#         best_params['objective'] = 'binary:logistic'
#     else:
#         raise ValueError("Task must be 'binary' or 'multiclass'.")

#     # -------------------------------
#     # Create DMatrix
#     # -------------------------------
#     dtrain = xgb.DMatrix(X_train, label=y_train, weight=sample_weights)
    
#     evals = [(dtrain, 'train')]
#     if X_val is not None and y_val is not None:
#         if task.lower() == "multiclass":
#             dval = xgb.DMatrix(X_val, label=y_val)  # same sample weights can be added optionally
#         else:
#             dval = xgb.DMatrix(X_val, label=y_val)
#         evals.append((dval, 'valid'))

#     # -------------------------------
#     # Determine number of boosting rounds
#     # -------------------------------
#     num_boost_round = best_iteration or best_params.get('num_boost_round', 1000)
    
#     # -------------------------------
#     # Train model
#     # -------------------------------
#     model = xgb.train(
#         params=best_params,
#         dtrain=dtrain,
#         num_boost_round=num_boost_round,
#         evals=evals,
#         verbose_eval=100,
#         early_stopping_rounds=100 if X_val is not None else None
#     )
    
#     return model