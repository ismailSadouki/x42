import optuna
import time

def run_optuna(objective, cfg, n_trials=None, use_median_pruner=False, warmup_steps=2 ):
    """
    Create and run an Optuna study with TPESampler + SuccessiveHalvingPruner.
    Returns: study object, elapsed time
    """
    start = time.time()
    startup_trials = getattr(cfg, "STARTUP_TRIALS", 20)
    if use_median_pruner:
        startup_trials = 15 # Aggressive start for MedianPruner


    sampler = optuna.samplers.TPESampler(
        n_startup_trials=startup_trials,
        multivariate=True,
        seed=cfg.RANDOM_STATE
    )
    

    if use_median_pruner:

        pruner = optuna.pruners.MedianPruner( # 2. Switch to MedianPruner for simpler "kill" logic
            n_startup_trials=startup_trials,
            n_warmup_steps=warmup_steps, # lower for large data
            interval_steps=1
        )
    else:
        
        pruner = optuna.pruners.SuccessiveHalvingPruner(
            min_resource=warmup_steps, # Start checking after Fold 2
            reduction_factor=3, # 👈 Change from 4 to 2 to be less "brutal"
            min_early_stopping_rate=0 # 👈 Change to 0 to check at every fold after the first rung
        )

    
    study = optuna.create_study(
        direction='maximize' if cfg.MAXIMIZE_METRIC else 'minimize',
        sampler=sampler,
        pruner=pruner
    )
    
    study.optimize(objective, n_trials=n_trials or cfg.N_TRIALS,timeout=getattr(cfg, "TIMEOUT", None), show_progress_bar=True)
    
    end = time.time()


    pruned_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]
    complete_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    
    print("\n" + "="*50)
    print(f"OPTUNA REPORT")
    print("="*50)
    print(f"Total Trials: {len(study.trials)}")
    print(f"Complete Trials: {len(complete_trials)}")
    print(f"Pruned Trials: {len(pruned_trials)} ({(len(pruned_trials)/len(study.trials))*100:.1f}%)")
    print(f"Optuna finished in: {end - start:.2f} seconds")
    print(f"Best Params: {study.best_params}")
    print(f"The best mean CV score across all trials: {study.best_value:.5f}")
    print("="*50 + "\n")


    
    return study

