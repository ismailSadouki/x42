import optuna
import time

def run_optuna(objective, cfg, n_trials=None):
    """
    Create and run an Optuna study with TPESampler + SuccessiveHalvingPruner.
    Returns: study object, elapsed time
    """
    start = time.time()
    
    sampler = optuna.samplers.TPESampler(
        n_startup_trials=cfg.STARTUP_TRIALS, # first 30 trials are purely random (explore full space)
        multivariate=True,
        seed=cfg.RANDOM_STATE
    )
    
    pruner = optuna.pruners.SuccessiveHalvingPruner(
        min_resource=2,
        reduction_factor=4,
        min_early_stopping_rate=1
    )
    
    study = optuna.create_study(
        direction='maximize' if cfg.MAXIMIZE_METRIC else 'minimize',
        sampler=sampler,
        pruner=pruner
    )
    
    study.optimize(objective, n_trials=n_trials or cfg.N_TRIALS, timeout=cfg.TIMEOUT)
    
    end = time.time()
    print(f"Optuna finished in {end - start:.2f} seconds")
    print("Best params:", study.best_params)
    print("The best mean CV score across all trials:", study.best_value)
    
    return study
