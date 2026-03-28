# src/config.py
import os

def find_project_root(marker_files=("train.csv", "sample_submission.csv", "src")):
    cwd = os.getcwd()
    while True:
        if any(os.path.exists(os.path.join(cwd, f)) for f in marker_files):
            return cwd
        parent = os.path.dirname(cwd)
        if parent == cwd:
            raise FileNotFoundError(f"Could not find project root with marker files: {marker_files}")
        cwd = parent



# Random seed
SEED = 42
N_FOLDS = 5

# Paths
DATA_DIR = "../data"
TRAIN_PATH = f"{DATA_DIR}/train.csv"
TEST_PATH = f"{DATA_DIR}/test.csv"
SUB_PATH = f"{DATA_DIR}/sample_submission.csv"

# Model defaults
# EARLY_STOPPING = 200
# N_ESTIMATORS = 5000



# src/config.py
class BaseConfig:
    PROJECT_ROOT = find_project_root()
    EXPERIMENTS_DIR = os.path.join(PROJECT_ROOT, "outputs", "experiments")
    DATA_DIR = os.path.join(PROJECT_ROOT, "data")
    SUB_PATH = os.path.join(PROJECT_ROOT, "sample_submission.csv")
    
    


    TARGET = "Target"
    ID = "id"
    SUBMIT_PROBABILITIES = False
    TARGET_MAPPER = {
        'A' : 0,
        'B' : 1,
        'C' : 2,
        }
    
    N_SPLITS = 5

    # Training
    NUM_BOOST_ROUND = 4000
    EARLY_STOPPING = 100

    # Optuna
    N_TRIALS = 100
    TIMEOUT = 3600
    STARTUP_TRIALS = 30

    KAGGLE_EVAL = "accuracy" # mpa@3/accuracy
    USE_POSTPROCESSING = True
    # Model
    TASK = "multiclass" # multiclass/binary
    METRIC = "accuracy" # auc/accuracy/multi_logloss
    MAXIMIZE_METRIC = False # minimize logloss False, True for auc/MP@3
    IS_UNBALANCE = False


    # Mapping to library-specific parameters
    # binary and auc
    # LIB_PARAMS = {
    #     "lightgbm": {"objective": "binary", "metric": "auc"},
    #     "xgboost": {"objective": "binary:logistic", "eval_metric": "auc"},
    #     "catboost": {"objective": "Logloss", "eval_metric": "AUC"}
    # }


    # seeds
    RANDOM_STATE = 42
    FEATURE_FRACTION_SEED = 42
    BAGGING_SEED = 42
    DATA_RANDOM_SEED = 42


    LARGE_DATASET_THRESHOLD = 500000  # adjust based on your RAM

    @classmethod
    def summary(cls):
        print(f"Target: {cls.TARGET}, Splits: {cls.N_SPLITS}, Boosting: {cls.BOOSTING_TYPE}")
        print(f"Project root: {cls.PROJECT_ROOT}")


class BinaryLoglossConfig(BaseConfig):
    OBJECTIVE = "binary"
    METRIC = "binary_logloss"
    LIB_PARAMS = {
        "lightgbm": {"objective": "binary", "metric": "auc"},
        "xgboost": {"objective": "binary:logistic", "eval_metric": "logloss"},
        "catboost": {"objective": "Logloss", "eval_metric": "Logloss"},
        "rf": {"objective": "binary", "metric": "oob"},
        "histgb": {"objective": "binary", "metric": "log_loss"},
    }

class MultiClassConfig(BaseConfig):
    OBJECTIVE = "multiclass"
    METRIC = "accuracy" # multi_logloss/accuracy 
    LIB_PARAMS = {
        "lightgbm": {"objective": "multiclass", "metric": "multi_logloss"}, # metric:multi_logloss|| objective: multiclass → better global ranking | multiclassova → better rare class detection Or Train BOTH and Blend predictions
        "xgboost": {"objective": "multi:softprob", "eval_metric": "mlogloss"}, # there is also multi_output_tree u could try it latter
        "catboost": {"objective": "multiclass", "eval_metric": "Accuracy"}, # 👉 ⚠️ Change to: "eval_metric": "MultiClass" Or "eval_metric": "MultiClassLogloss"
        "rf": {"objective": "multiclass", "metric": "oob"},
        "histgb": {"objective": "multiclass", "metric": "log_loss"},

    
    }

# 👉 post-processing matters more than training metric for accuaracy
# search about label smoothing (during training)



# ⚔️ Which is better for Kaggle?
# 🏆 If metric = log_loss:

# 👉 BEST:

# logit calibration ✅

# blending ✅

# stacking ✅

# ❌ threshold optimization

# 🏆 If metric = accuracy / F1:

# 👉 BEST:

# threshold optimization ✅

# class weights ✅

class SmallTraining:
    NUM_BOOST_ROUND = 200
    N_TRIALS = 10


class FullTraining:
    NUM_BOOST_ROUND = 5000
    N_TRIALS = 100


class LGBConfig(BaseConfig):
    OBJECTIVE = "binary"
    METRIC = "auc"
    BOOSTING_TYPE = "gbdt"

class XGBConfig(BaseConfig):
    OBJECTIVE = "binary:logistic"
    METRIC = "auc"
    N_ESTIMATORS = 1000

## ACTIVE CONFIG
class Config(MultiClassConfig, FullTraining):
    pass


# can do cfg = Config(task="binary", mode="small", model="lgbm")



















#-------------------------------------
# Optimize Ensemble Weights
# -----------------------------------
class OptunaWeights:
    def __init__(self, random_state, n_trials=5000):
        self.study = None
        self.weights = None
        self.random_state = random_state
        self.n_trials = n_trials

    def _objective(self, trial, y_true, y_preds):
        # Define the weights for the predictions from each model
        weights = [trial.suggest_float(f"weight{n}", 0, 1) for n in range(len(y_preds))]

        # Calculate the weighted prediction
        weighted_pred = np.average(np.array(y_preds).T, axis=1, weights=weights)

        auc_score = roc_auc_score(y_true, weighted_pred)
        log_loss_score=log_loss(y_true, weighted_pred)
        return auc_score#/log_loss_score

    def fit(self, y_true, y_preds):
        optuna.logging.set_verbosity(optuna.logging.ERROR)
        sampler = optuna.samplers.CmaEsSampler(seed=self.random_state)
        pruner = optuna.pruners.HyperbandPruner()
        self.study = optuna.create_study(sampler=sampler, pruner=pruner, study_name="OptunaWeights", direction='maximize')
        objective_partial = partial(self._objective, y_true=y_true, y_preds=y_preds)
        self.study.optimize(objective_partial, n_trials=self.n_trials)
        self.weights = [self.study.best_params[f"weight{n}"] for n in range(len(y_preds))]

    def predict(self, y_preds):
        assert self.weights is not None, 'OptunaWeights error, must be fitted before predict'
        weighted_pred = np.average(np.array(y_preds).T, axis=1, weights=self.weights)
        return weighted_pred

    def fit_predict(self, y_true, y_preds):
        self.fit(y_true, y_preds)
        return self.predict(y_preds)
    
    def weights(self):
        return self.weights
    
# -----------------------
# Model Fit
# -------------------------

# class Classifier:
#     def __init__(self, n_estimators=100, device="cpu", random_state=0):
#         self.n_estimators = n_estimators
#         self.device = device
#         self.random_state = random_state
#         self.models = self._define_model()
#         self.len_models = len(self.models)
        
#     def _define_model(self):
#         xgb_params = {
#             'n_estimators': self.n_estimators,
#             'learning_rate': 0.1,
#             'max_depth': 4,
#             'subsample': 0.8,
#             'colsample_bytree': 0.1,
#             'n_jobs': -1,
#             'eval_metric': 'logloss',
#             'objective': 'binary:logistic',
#             'tree_method': 'hist',
#             'verbosity': 0,
#             'random_state': self.random_state,
# #             'class_weight':class_weights_dict,
#         }
#         if self.device == 'gpu':
#             xgb_params['tree_method'] = 'gpu_hist'
#             xgb_params['predictor'] = 'gpu_predictor'
            
#         xgb_params2=xgb_params.copy() 
#         xgb_params2['subsample']= 0.5
#         xgb_params2['max_depth']=9
#         xgb_params2['learning_rate']=0.045
#         xgb_params2['colsample_bytree']=0.3

#         xgb_params3=xgb_params.copy() 
#         xgb_params3['subsample']= 0.6
#         xgb_params3['max_depth']=6
#         xgb_params3['learning_rate']=0.02
#         xgb_params3['colsample_bytree']=0.7      

#         xgb_params4=xgb_params.copy() 
#         xgb_params4['subsample']= 0.5943421542786502
#         xgb_params4['max_depth']=6
#         xgb_params4['learning_rate']=0.109
#         xgb_params4['colsample_bytree']=0.5595039093313848
#         lgb_params = {
#             'n_estimators': self.n_estimators,
#             'max_depth': 8,
#             'learning_rate': 0.02,
#             'subsample': 0.20,
#             'colsample_bytree': 0.56,
#             'reg_alpha': 0.25,
#             'reg_lambda': 5e-08,
#             'objective': 'binary',
#             'boosting_type': 'gbdt',
#             'device': self.device,
#             'random_state': self.random_state,
# #             'class_weight':class_weights_dict,
#         } 
#         lgb_params2 = {
#             'n_estimators': self.n_estimators,
#             'max_depth': 5,
#             'learning_rate': 0.015,
#             'subsample': 0.50,
#             'colsample_bytree': 0.1,
#             'reg_alpha': 0.07608657669988828,
#             'reg_lambda': 0.2255036530113883,
#             'objective': 'binary',
#             'boosting_type': 'gbdt',
#             'device': self.device,
#             'random_state': self.random_state,
#         }
#         lgb_params3=lgb_params.copy()  
#         lgb_params3['subsample']=0.9
#         lgb_params3['reg_lambda']=0.3461495211744402
#         lgb_params3['reg_alpha']=0.3095626288582237
#         lgb_params3['max_depth']=8
#         lgb_params3['learning_rate']=0.007
#         lgb_params3['colsample_bytree']=0.5

#         lgb_params4=lgb_params2.copy()  
#         lgb_params4['subsample']=0.3
#         lgb_params4['reg_lambda']=0.49406951573373614
#         lgb_params4['reg_alpha']=0.16269100796945424
#         lgb_params4['max_depth']=9
#         lgb_params4['learning_rate']=0.117
#         lgb_params4['colsample_bytree']=0.3

#         cb_params = {
#             'iterations': self.n_estimators,
#             'depth': 13,
#             'learning_rate': 0.015,
#             'l2_leaf_reg': 0.5,
#             'random_strength': 0.1,
#             'max_bin': 200,
#             'od_wait': 65,
#             'one_hot_max_size': 50,
#             'grow_policy': 'Depthwise',
#             'bootstrap_type': 'Bernoulli',
#             'od_type': 'Iter',
#             'eval_metric': 'AUC',
#             'loss_function': 'Logloss',
#             'task_type': self.device.upper(),
#             'random_state': self.random_state,
#         }
#         cb_sym_params = cb_params.copy()
#         cb_sym_params['grow_policy'] = 'SymmetricTree'
#         cb_loss_params = cb_params.copy()
#         cb_loss_params['grow_policy'] = 'Lossguide'
        
#         cb_params2=  cb_params.copy()
#         cb_params2['learning_rate']=0.01
#         cb_params2['depth']=8

#         cb_params3={
#             'iterations': self.n_estimators,
#             'random_strength': 0.5783342241486167, 
#             'one_hot_max_size': 10, 
#             'max_bin': 150, 
#             'learning_rate': 0.177, 
#             'l2_leaf_reg': 0.705662073971363, 
#             'grow_policy': 'SymmetricTree', 
#             'depth': 5, 
#             'max_bin': 200,
#             'od_wait': 65,
#             'bootstrap_type': 'Bayesian',
#             'od_type': 'Iter',
#             'eval_metric': 'AUC',
#             'loss_function': 'Logloss',
#             'task_type': self.device.upper(),
#             'random_state': self.random_state,
#         }
#         cb_params4=  cb_params.copy()
#         cb_params4['learning_rate']=0.01
#         cb_params4['depth']=12
#         dt_params= {'min_samples_split': 30, 'min_samples_leaf': 10, 'max_depth': 8, 'criterion': 'gini'}
        
#         models = {
#             'xgb': xgb.XGBClassifier(**xgb_params),
#             'xgb2': xgb.XGBClassifier(**xgb_params2),
#             'xgb3': xgb.XGBClassifier(**xgb_params3),
#             'xgb4': xgb.XGBClassifier(**xgb_params4),
#             'lgb': lgb.LGBMClassifier(**lgb_params),
#             'lgb2': lgb.LGBMClassifier(**lgb_params2),
#             'lgb3': lgb.LGBMClassifier(**lgb_params3),
#             'lgb4': lgb.LGBMClassifier(**lgb_params4),
#             'cat': CatBoostClassifier(**cb_params),
# #             'cat2': CatBoostClassifier(**cb_params2),
#             'cat3': CatBoostClassifier(**cb_params3),
# #             'cat4': CatBoostClassifier(**cb_params4),
#             "cat_sym": CatBoostClassifier(**cb_sym_params),
#             "cat_loss": CatBoostClassifier(**cb_loss_params),
#             'hist_gbm' : HistGradientBoostingClassifier (max_iter=300, learning_rate=0.001,  max_leaf_nodes=80,
#                                                          max_depth=6,random_state=self.random_state),#class_weight=class_weights_dict, 
#             'gbdt': GradientBoostingClassifier(max_depth=6,  n_estimators=1000,random_state=self.random_state),
#             'lr': LogisticRegression(),
#             'rf': RandomForestClassifier(max_depth= 9,max_features= 'auto',min_samples_split= 10,
#                                                           min_samples_leaf= 4,  n_estimators=500,random_state=self.random_state),
# #             'svc': SVC(gamma="auto", probability=True),
# #             'knn': KNeighborsClassifier(n_neighbors=5),
# #             'mlp': MLPClassifier(random_state=self.random_state, max_iter=1000),
#             'etr':ExtraTreesClassifier(min_samples_split=55, min_samples_leaf= 15, max_depth=10,
#                                        n_estimators=200,random_state=self.random_state),
# #             'dt' :DecisionTreeClassifier(**dt_params,random_state=self.random_state),
# #             'ada': AdaBoostClassifier(random_state=self.random_state),
#             'ann':ann,
                                       
#         }
#         return models

# kfold = True
# n_splits = 1 if not kfold else 5
# random_state = 2023
# random_state_list = [42] # used by split_data [71]
# n_estimators = 9999 # 9999
# early_stopping_rounds = 300
# verbose = False

# splitter = Splitter(kfold=kfold, n_splits=n_splits)

# # Initialize an array for storing test predictions
# test_predss = np.zeros(X_test.shape[0])
# ensemble_score = []
# weights = []
# trained_models = {'xgb':[], 'lgb':[]}

    
# for i, (X_train_, X_val, y_train_, y_val) in enumerate(splitter.split_data(X_train, y_train, random_state_list=random_state_list)):
#     n = i % n_splits
#     m = i // n_splits
            
#     # Get a set of Regressor models
#     classifier = Classifier(n_estimators, device, random_state)
#     models = classifier.models
    
#     # Initialize lists to store oof and test predictions for each base model
#     oof_preds = []
#     test_preds = []
    
#     # Loop over each base model and fit it to the training data, evaluate on validation data, and store predictions
#     for name, model in models.items():
#         if ('cat' in name) or ("lgb" in name) or ("xgb" in name):
#             if 'lgb' == name: #categorical_feature=cat_features
#                 model.fit(X_train_, y_train_, eval_set=[(X_val, y_val)],#,categorical_feature=cat_features,
#                           early_stopping_rounds=early_stopping_rounds, verbose=verbose)
#             elif 'cat' ==name:
#                 model.fit(X_train_, y_train_, eval_set=[(X_val, y_val)],#cat_features=cat_features,
#                           early_stopping_rounds=early_stopping_rounds, verbose=verbose)
#             else:
#                 model.fit(X_train_, y_train_, eval_set=[(X_val, y_val)], early_stopping_rounds=early_stopping_rounds, verbose=verbose)
#         elif name in 'ann':
#             model.fit(X_train_, y_train_, validation_data=(X_val, y_val),batch_size=4, epochs=5,verbose=verbose)
#         else:
#             model.fit(X_train_, y_train_)
        
#         if name in 'ann':
#             test_pred = np.array(model.predict(X_test))[:, 0]
#             y_val_pred = np.array(model.predict(X_val))[:, 0]
#         else:
#             test_pred = model.predict_proba(X_test)[:, 1]
#             y_val_pred = model.predict_proba(X_val)[:, 1]

#         score = roc_auc_score(y_val, y_val_pred)
# #         score = accuracy_score(y_val, acc_cutoff_class(y_val, y_val_pred))

#         print(f'{name} [FOLD-{n} SEED-{random_state_list[m]}] ROC AUC score: {score:.5f}')
        
#         oof_preds.append(y_val_pred)
#         test_preds.append(test_pred)
        
#         if name in trained_models.keys():
#             trained_models[f'{name}'].append(deepcopy(model))
#     # Use Optuna to find the best ensemble weights
#     optweights = OptunaWeights(random_state=random_state)
#     y_val_pred = optweights.fit_predict(y_val.values, oof_preds)
    
#     score = roc_auc_score(y_val, y_val_pred)
# #     score = accuracy_score(y_val, acc_cutoff_class(y_val, y_val_pred))
#     print(f'Ensemble [FOLD-{n} SEED-{random_state_list[m]}] ------------------>  ROC AUC score {score:.5f}')
#     ensemble_score.append(score)
#     weights.append(optweights.weights)
    
#     test_predss += optweights.predict(test_preds) / (n_splits * len(random_state_list))
    
#     gc.collect()







# ----------------------------
# submission file
# ---------------------------


# def submit_file(model, X_test, filename = 'submission.csv'):
#     preds = model.predict_proba(X_test)
#     predicted_prob = [pred[1] for pred in preds]
#     sub_df['Machine failure'] = predicted_prob
#     sub_df.to_csv(filename, index = False)
#     print(f'{blu}File successfully created with name {filename}')

