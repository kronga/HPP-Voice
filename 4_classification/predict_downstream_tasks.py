import json
import logging
import math
import os
import random
import warnings
from collections import namedtuple
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

import xgboost as xgb
import lightgbm as lgb
import optuna
from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    auc,
    f1_score,
    mean_squared_error,
    precision_recall_curve,
    r2_score,
    roc_auc_score,
    mean_absolute_error,
)
from sklearn.model_selection import KFold, StratifiedKFold, GroupKFold, cross_val_score, StratifiedGroupKFold
from statistics import LinearRegression

from LabQueue.qp import qp, fakeqp
from LabUtils.addloglevels import sethandlers

import yaml
cfg = yaml.safe_load(open("src/models/config_predict_age.yaml"))

def log_configuration(config):
    """Log the configuration in a readable format."""
    logging.info("=== CONFIGURATION ===")
    for section, values in config.items():
        if isinstance(values, dict):
            logging.info(f"\n{section}:")
            for key, value in values.items():
                if isinstance(value, dict):
                    logging.info(f"  {key}:")
                    for subkey, subvalue in value.items():
                        logging.info(f"    {subkey}: {subvalue}")
                else:
                    logging.info(f"  {key}: {value}")
        else:
            logging.info(f"{section}: {values}")
    logging.info("===================")

# Configuration Parameters
# ======================
# Paths
PATH_FOR_MEDICAL_CONDITIONS = cfg["PATH_FOR_MEDICAL_CONDITIONS"]
RealDeepFolderPath = cfg["RealDeepFolderPath"]
DeepFolderPath = cfg["DeepFolderPath"]
OUTPUT_PATH = cfg["OUTPUT_PATH"]
MEDICAL_FILE_PATH = cfg["MEDICAL_FILE_PATH"]
OPTUNA_STORAGE = JournalStorage(JournalFileBackend(cfg["OPTUNA_JOURNAL_PATH"]))
MEAN_DIR = Path(cfg["OUTPUT_PATH"]) / "mean_embeddings"
SUBJECT_DETAILS_TABLE = cfg["SUBJECT_DETAILS_TABLE"]
DATA_DIR = cfg["DATA_DIR"]
# Logging Configuration
LOG_DIR = cfg["LOG_DIR"]
os.makedirs(LOG_DIR, exist_ok=True)
LOG_FILE = Path(cfg["LOG_DIR"]) / f"run_{datetime.now():%Y%m%d_%H%M%S}.log"

warnings.filterwarnings("ignore", message="No further splits with positive gain")
warnings.filterwarnings("ignore", 
                       message="A value is trying to be set on a copy of a DataFrame or Series through chained assignment using an inplace method",
                       category=FutureWarning)

class LightGBMWarningFilter(logging.Filter):
    def filter(self, record):
        # Check if the message contains the unwanted warning text.
        # If so, return False to ignore the record.
        return "[LightGBM] [Warning] No further splits with positive gain" not in record.getMessage()

class PandasFutureWarningFilter(logging.Filter):
    def filter(self, record):
        # Check if the message contains the pandas FutureWarning about inplace methods
        return "FutureWarning: A value is trying to be set on a copy of a DataFrame or Series through chained assignment using an inplace method" not in record.getMessage()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler()  # This will also print to console
    ]
)

for handler in logging.getLogger().handlers:
    handler.addFilter(LightGBMWarningFilter())
    handler.addFilter(PandasFutureWarningFilter())

# Log the configuration at the start
log_configuration(cfg)

# Model Parameters
SEED = cfg["SEED"]  
NUM_THREAD = cfg["NUM_THREAD"]
NUM_TRIALS = cfg["NUM_TRIALS"]
MODEL_TYPE = cfg["MODEL_TYPE"]  
N_SPLITS = cfg["N_SPLITS"]  
BALANCE_CLASSES = cfg["BALANCE_CLASSES"]
COMBINE_WITH_BASELINE = cfg["COMBINE_WITH_BASELINE"]
GENDER_SPECIFIC = cfg["GENDER_SPECIFIC"]
DEBUG_MODE = cfg["DEBUG_MODE"] 
FORCE_NEW_STUDIES = cfg["FORCE_NEW_STUDIES"] 

# Initialize random seeds for reproducibility
random.seed(SEED)
np.random.seed(SEED)
os.environ['PYTHONHASHSEED'] = str(SEED)

# Embedding Configurations
EMBEDDING_CONFIGS = cfg["EMBEDDING_CONFIGS"]
   

# Debug Mode Settings
DEBUG_EMBEDDINGS = cfg["DEBUG_EMBEDDINGS"]
DEBUG_TARGETS = cfg["DEBUG_TARGETS"]


TARGET_NAMES_CALCULATED = cfg["TARGET_NAMES_CALCULATED"]

CURATED_CONDITIONS = cfg["CURATED_CONDITIONS"]

QUEUE_PARAMS = cfg["QUEUE_PARAMS"]

# Pipeline execution control flags
RUN_HPO_PHASE = cfg["RUN_HPO_PHASE"]       # Whether to run hyperparameter optimization
RUN_CV_PHASE = cfg["RUN_CV_PHASE"]       # Whether to run cross-validation with optimized parameters
RUN_TEST_PHASE = cfg["RUN_TEST_PHASE"]     # Whether to evaluate on test set
LOAD_HPO_RESULTS = cfg["LOAD_HPO_RESULTS"]  # Whether to load previous HPO results when skipping HPO phase

Result = namedtuple(
    "Result", ["target", "embedding", "test_auc", "test_f1", "test_auprc", "prev"]
)


def check_classes(y):
    """Check if there are at least two classes present in the dataset."""
    return len(np.unique(y)) >= 2

def age_to_bins(y: np.ndarray, width: int = 2) -> np.ndarray:
    """
    Return an integer label per sample such that all ages that fall into the
    same <width>-year window share the same label.
    Example (width=2):  23.4 ⟶ 11,  24.8 ⟶ 12,  25.9 ⟶ 12, …
    The labels are ONLY used for CV splitting – the model still sees
    the *continuous* age values.
    """
    y = np.asarray(y, dtype=float)
    return (y // width).astype(int)


def build_estimator(trial, model_type, is_regression, is_balance, scale_pos_weight=None, seed_num=SEED):
    """Build a model estimator with hyperparameters from Optuna trial.
    
    Args:
        trial: Optuna trial object for suggesting hyperparameters
        model_type: Type of model to build ("lightgbm", "xgboost", or "logistic regression")
        is_regression: Whether this is a regression task
        is_balance: Whether to apply class balancing
        scale_pos_weight: Weight for positive class (for classification tasks)
        seed_num: Random seed for reproducibility
        
    Returns:
        sklearn-compatible estimator configured with trial-suggested hyperparameters
    """
    if is_regression:
        if model_type.lower() == "linear regression":
            params = {}
            model = LinearRegression(**params)
        elif model_type.lower() == "xgboost":
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
                "max_depth": trial.suggest_int("max_depth", 3, 10),
                "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.1, log=True),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
                "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
                "gamma": trial.suggest_float("gamma", 1e-8, 1.0, log=True),
                "random_state": seed_num,
            }
            model = xgb.XGBRegressor(**params)
        elif model_type.lower() == "lightgbm":
            params = {
                "max_depth": trial.suggest_int("max_depth", 3, 9),
                "learning_rate": trial.suggest_float("learning_rate", 0.001, 0.1, log=True),
                "n_estimators": trial.suggest_int("n_estimators", 500, 2000, step=500),
                "subsample": trial.suggest_float("subsample", 0.6, 0.8, step=0.1),
                "feature_fraction": trial.suggest_float("feature_fraction", 0.5, 0.9, step=0.1),
                "random_state": seed_num,
            }
            max_leaves = 2 ** params["max_depth"]
            params["num_leaves"] = trial.suggest_int("num_leaves", 8, min(350, max_leaves))
            params["reg_alpha"] = 1e-4
            params["reg_lambda"] = 1e-4
            params["min_split_gain"] = 1e-5
            params["num_threads"] = NUM_THREAD

            model = lgb.LGBMRegressor(**params)
    else:
        if model_type.lower() == "logistic regression":
            params = {
                "C": trial.suggest_float("C", 1e-4, 1e4, log=True),
                "penalty": trial.suggest_categorical("penalty", ["l1", "l2"]),
                "class_weight": "balanced" if is_balance else None,
                "solver": "liblinear",
                "random_state": seed_num,
            }
            model = LogisticRegression(**params)
        elif model_type.lower() == "xgboost":
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
                "max_depth": trial.suggest_int("max_depth", 3, 10),
                "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.1, log=True),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
                "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
                "gamma": trial.suggest_float("gamma", 1e-8, 1.0, log=True),
                "random_state": seed_num,
            }
            if is_balance and scale_pos_weight is not None:
                params["scale_pos_weight"] = scale_pos_weight
            model = xgb.XGBClassifier(**params, use_label_encoder=False, eval_metric="auc")
        elif model_type.lower() == "lightgbm":
            params = {
                "max_depth": trial.suggest_int("max_depth", 3, 9),
                "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.1, log=True),
                "n_estimators": trial.suggest_int("n_estimators", 500, 1000, step=100),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0, step=0.1),
                "feature_fraction": trial.suggest_float("feature_fraction", 0.3, 0.9, step=0.1),
                "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 1.0, log=True),
                "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 1.0, log=True),
                "min_split_gain": trial.suggest_float("min_split_gain", 1e-8, 0.1, log=True),
                "random_state": seed_num,
            }
            max_leaves = 2 ** params["max_depth"]
            params["num_leaves"] = trial.suggest_int("num_leaves", 8, max_leaves)
            params["num_threads"] = NUM_THREAD
            if is_balance and scale_pos_weight is not None:
                params["scale_pos_weight"] = scale_pos_weight
            model = lgb.LGBMClassifier(**params)
    
    return model, params


def train_final_model(X_train, y_train, best_params, model_type, is_regression, is_balance=True, seed_num=SEED):
    """Train final model with best hyperparameters.
    
    Args:
        X_train: Training features
        y_train: Training target values
        best_params: Dictionary of best hyperparameters from HPO
        model_type: Type of model to train
        is_regression: Whether this is a regression task
        is_balance: Whether to apply class balancing
        seed_num: Random seed for reproducibility
        
    Returns:
        Trained model
    """
    logging.info(f"Training final {model_type} model with best parameters")
    # Validate inputs
    supported_regression_models = ["linear regression", "xgboost", "lightgbm"]
    supported_classification_models = ["logistic regression", "xgboost", "lightgbm"]
    model_type_lower = model_type.lower()

    if is_regression and model_type_lower not in supported_regression_models:
        logging.error(f"Unsupported regression model type: {model_type}")
        return None
    elif not is_regression and model_type_lower not in supported_classification_models:
        logging.error(f"Unsupported classification model type: {model_type}")
        return None
    
    # Convert inputs to numpy arrays
    X_train = np.array(X_train, copy=True)
    y_train = np.array(y_train, copy=True)
    
    if not is_regression:
        y_train = y_train.astype(int)
        # Check if binary classification
        unique_classes = np.unique(y_train)
        if len(unique_classes) != 2:
            logging.warning(f"Expected binary classification but found {len(unique_classes)} classes")
    
    # Calculate scale_pos_weight for classification
    scale_pos_weight = None
    if not is_regression and is_balance and len(np.unique(y_train)) == 2:
        n_neg = np.sum(y_train == 0)
        n_pos = np.sum(y_train == 1)
        if n_pos > 0:  # Avoid division by zero
            scale_pos_weight = round(n_neg / n_pos)
            logging.info(f"Class balance ratio (neg:pos): {scale_pos_weight}:1")

    # Build final model with best parameters
    params = best_params.copy() if best_params else {}

    try:
        if is_regression:
            if model_type_lower == "linear regression":
                model = LinearRegression()
            elif model_type_lower == "xgboost":
                params["random_state"] = seed_num
                model = xgb.XGBRegressor(**params)
            elif model_type_lower == "lightgbm":
                params["random_state"] = seed_num
                params["num_threads"] = NUM_THREAD
                model = lgb.LGBMRegressor(**params)
        else:
            if model_type_lower == "logistic regression":
                params["random_state"] = seed_num
                params["class_weight"] = "balanced" if is_balance else None
                params["solver"] = params.get("solver", "liblinear")
                model = LogisticRegression(**params)
            elif model_type_lower == "xgboost":
                params["random_state"] = seed_num
                if is_balance and scale_pos_weight:
                    params["scale_pos_weight"] = scale_pos_weight
                
                # Handle eval_metric based on params or set default
                if "eval_metric" not in params:
                    params["eval_metric"] = "auc"
                    
                model = xgb.XGBClassifier(**params)
            elif model_type_lower == "lightgbm":
                params["random_state"] = seed_num
                params["num_threads"] = NUM_THREAD
                if is_balance and scale_pos_weight:
                    params["scale_pos_weight"] = scale_pos_weight
                model = lgb.LGBMClassifier(**params)
        
        # Fit model
        model.fit(X_train, y_train)
        logging.info("Model training completed successfully")
        return model
    except Exception as e:
        logging.error(f"Error training model: {str(e)}")
        return None

def evaluate(model, X_test, y_test, is_regression):
    """Evaluate model on test data.
    
    Args:
        model: Trained model
        X_test: Test features
        y_test: Test target values
        is_regression: Whether this is a regression task
        
    Returns:
        Dictionary of evaluation metrics
    """
    # Convert inputs to numpy arrays
    X_test = np.array(X_test, copy=True)
    y_test = np.array(y_test, copy=True)
    
    if not is_regression:
        y_test = y_test.astype(int)
    
    result = {}
    
    try:
        if is_regression:
            y_pred = model.predict(X_test)
            result.update({
                "mse": mean_squared_error(y_test, y_pred),
                "r2": r2_score(y_test, y_pred),
                "rmse": np.sqrt(mean_squared_error(y_test, y_pred)),
                "mae": mean_absolute_error(y_test, y_pred),
                "predictions": y_pred,
                "true_values": y_test,
            })
        else:
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            binary_preds = (y_pred_proba >= 0.5).astype(int)
            precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
            result.update({
                "auc": roc_auc_score(y_test, y_pred_proba),
                "f1": f1_score(y_test, binary_preds),
                "auprc": auc(recall, precision),
                "accuracy": accuracy_score(y_test, binary_preds),
                "prevalence": y_test.mean(),
                "predictions": y_pred_proba,
                "true_values": y_test,
            })
        
        return result
    
    except Exception as e:
        logging.error(f"Error evaluating model: {str(e)}")
        return None

def run_hpo(X, y, model_type, is_balance=True, groups=None, n_splits=N_SPLITS, n_trials=NUM_TRIALS, seed_num=SEED, n_jobs=NUM_THREAD, study_name=None, force_new_studies=FORCE_NEW_STUDIES):
    """Run hyperparameter optimization with cross-validation handled by Optuna.
    
    Args:
        X: Features dataset
        y: Target values
        model_type: Type of model to optimize
        is_balance: Whether to apply class balancing
        groups: List of group IDs
        n_splits: Number of cross-validation splits
        n_trials: Number of trials to run
        seed_num: Random seed for reproducibility
        n_jobs: Number of parallel jobs
        study_name: Name of the study to use
        force_new_studies: Whether to force new studies
    Returns:
        Dictionary of best hyperparameters
    """
    logging.info(f"Starting HPO with {model_type} for {n_trials} trials using {n_splits}-fold CV")
    
    # Convert inputs to numpy arrays
    X = np.array(X)
    y = np.array(y)
    
    # Determine if it's a regression task
    is_regression = isinstance(y[0], (np.floating, float))
    
    if not is_regression:
        y = y.astype(int)
        if len(np.unique(y)) < 2:
            logging.warning("Not enough classes for classification")
            return None
    
    # Calculate scale_pos_weight for classification tasks
    scale_pos_weight = None
    if not is_regression and is_balance:
        scale_pos_weight = round(np.sum(y == 0) / np.sum(y == 1))
    
    # Set up cross-validation
    if is_regression:
        # --- regression WITH optional stratification --------------------------
        y_bins = age_to_bins(y, width=2)          # 2-year windows
        if groups is not None:
            cv = StratifiedGroupKFold(
                n_splits=n_splits, shuffle=True, random_state=seed_num)
            split_iter = cv.split(X, y_bins, groups)   # bins only guide the split
        else:
            cv = StratifiedKFold(
                n_splits=n_splits, shuffle=True, random_state=seed_num)
            split_iter = cv.split(X, y_bins)
    else:
        # original behaviour for classification
        cv = StratifiedGroupKFold(...) if groups is not None else StratifiedKFold(...)
        split_iter = cv.split(X, y, groups) if groups is not None else cv.split(X, y)


    logging.info(
        f"Using {cv.__class__.__name__} with {n_splits} splits"
        + (f" and {len(np.unique(groups))} groups" if groups is not None else ""))
    
    # Define the objective function for optimization
    def objective(trial):
        model, _ = build_estimator(
            trial=trial,
            model_type=model_type,
            is_regression=is_regression,
            is_balance=is_balance,
            scale_pos_weight=scale_pos_weight,
            seed_num=seed_num,
        )

        try:
            # unified call – `groups` is accepted even when it is None
            scores = cross_val_score(
                model,
                X,
                y,
                groups=groups,
                cv=list(split_iter),
                scoring="neg_mean_absolute_error" if is_regression else "roc_auc",
                n_jobs=1,
            )

            mean_score = scores.mean()
            # Optuna always *minimises* the objective
            if is_regression:
            # cross_val_score gave NEGATIVE MSE  →  flip sign to get real (positive) MSE
                return -mean_score          # Optuna will MINIMISE it
            else:
                return  mean_score     # both negate

        except Exception as e:
            logging.error(f"Error in HPO objective: {e}")
            # signal catastrophic failure in a way Optuna understands
            return float("inf")  if is_regression else float("-inf")
        
    # Create study with appropriate direction
    study_direction = "maximize" if not is_regression else "minimize"
    study = get_or_create_study(study_name=study_name, direction=study_direction, seed=seed_num, force_new=force_new_studies)

    # Run optimization with multiprocessing
    study.optimize(
        objective, 
        n_trials=n_trials, 
        n_jobs=n_jobs,
        timeout=None,
        catch=(Exception,),
        callbacks=[lambda study, trial: logging.info(f"Trial {trial.number} finished with value: {trial.value}")]
    )
    
    # Get best parameters
    best_params = study.best_params
    best_value = study.best_value
    
    logging.info(f"HPO completed. Best score: {best_value:.4f}")
    logging.info(f"Best parameters: {best_params}")
    
    # Save best parameters to file
    best_params_with_meta = {
        "best_params": best_params,
        "best_value": float(best_value),
        "model_type": model_type,
        "is_regression": is_regression,
        "n_trials": n_trials,
        "n_splits": n_splits,
        "seed": seed_num,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    
    return best_params_with_meta

def run_cv_for_config(config_key,
                      cfg_dict,
                      best_params,
                      model_type,
                      n_splits,
                      seed_num,
                      is_balance):
    """
    *Worker function* - performs the per config CV loop and
    returns the `avg_result` dict that the caller used to append
    into `cv_results`.
    """
    X_train         = cfg_dict['X_train']
    y_train         = cfg_dict['y_train']
    target          = cfg_dict['target']
    embedding       = cfg_dict['embedding']
    gender_label    = cfg_dict['gender_label']
    is_regression   = cfg_dict['is_regression']
    original_idx    = cfg_dict['original_train_indices']
    condition_dir   = cfg_dict['condition_dir']

    combined_path = condition_dir / f"cv_combined_predictions_{embedding}_{seed_num}.csv"
    if combined_path.exists():
        logging.info(f"[SKIP] {config_key} - {combined_path.name} already present")
        return None
    
    group_ids = [idx.split('_')[0] for idx in X_train.index]
    if is_regression:
        y_bins = age_to_bins(y_train, width=2)
        if group_ids is None:      # usual case
            cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed_num)
            cv_iterator = cv.split(X_train, y_bins)
        else:
            cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed_num)
            cv_iterator = cv.split(X_train, y_bins, groups=group_ids)
    else:                      # classification → we can stratify *and* group
        cv = StratifiedGroupKFold(
            n_splits=n_splits,
            shuffle=True,           # shuffle keeps fold sizes balanced
            random_state=seed_num   # reproducible splits
        )
        cv_iterator = cv.split(X_train, y_train, groups=group_ids)

    
    # results containers
    fold_results, all_preds, all_truths, all_fold, all_indices = \
        [], [], [], [], []


    for fold_idx, (tr, va) in enumerate(cv_iterator):
        X_tr, X_va = X_train.iloc[tr], X_train.iloc[va]
        y_tr, y_va = y_train.iloc[tr], y_train.iloc[va]

        model = train_final_model(X_tr, y_tr, best_params,
                                  model_type, is_regression,
                                  is_balance, seed_num)
        if model is None:
            logging.warning(f"Model train failed on {config_key} fold {fold_idx}")
            continue

        eval_ = evaluate(model, X_va, y_va, is_regression)
        if eval_ is None:
            logging.warning(f"Eval failed on {config_key} fold {fold_idx}")
            continue

        fold_res = {
            "Target": target,
            "Embedding": embedding,
            "Model Type": model_type,
            "Is Regression": is_regression,
            "Fold": fold_idx,
            "Best Parameters": best_params,
        }
        if gender_label:
            fold_res["Gender"] = gender_label
        for m, v in eval_.items():
            if m not in ("predictions", "true_values"):
                fold_res[m] = v
        fold_results.append(fold_res)

        # collect predictions (needed by caller for CSV/NPZ saving later)
        idx_va = original_idx[va]
        all_preds.extend(eval_["predictions"])
        all_truths.extend(eval_["true_values"])
        all_indices.extend(idx_va)
        all_fold.extend([fold_idx] * len(idx_va))


    pd.DataFrame({
        'index': all_indices,
        'true_values': all_truths,
        'predictions': all_preds,
        'fold': all_fold
    }).to_csv(combined_path, index=False)

    # fold‑level results CSV
    pd.DataFrame(fold_results).to_csv(os.path.join(
        condition_dir, f"cv_fold_results_{embedding}_{seed_num}.csv"),
        index=False)

    if not fold_results:
        return None                       # propagate failure to caller

    avg_result = {
        "Target": target,
        "Embedding": embedding,
        "Model Type": model_type,
        "Is Regression": is_regression,
        "Number of Folds": len(fold_results),
        "Evaluation": "Cross-Validation",
        "Best Parameters": best_params,
    }
    if gender_label:
        avg_result["Gender"] = gender_label

    for metric in fold_results[0]:
        if metric in ('Target', 'Embedding', 'Model Type', 'Is Regression',
                      'Fold', 'Gender', 'Best Parameters', 'Evaluation'):
            continue
        vals = [r[metric] for r in fold_results if metric in r]
        if vals and all(isinstance(v, (int, float)) for v in vals):
            avg_result[f"{metric}_mean"] = np.mean(vals)
            avg_result[f"{metric}_std"]  = np.std(vals)

    return avg_result

def run_test_for_config(config_key,
                        cfg_dict,
                        best_params,
                        model_type,
                        gender_specific,
                        is_balance,
                        test_ids,
                        dataframe,
                        seed_num):
    """
    Performs the old Phase‑3 per‑config test evaluation and returns
    the `test_result` dict.  Designed to be called inside a qp worker.
    """
    import os, numpy as np, pandas as pd, logging

    target        = cfg_dict['target']
    embedding     = cfg_dict['embedding']
    gender_label  = cfg_dict['gender_label']
    is_regression = cfg_dict['is_regression']
    condition_dir = cfg_dict['condition_dir']

    # full train / (cached) test matrices from Phase‑1
    X_train = cfg_dict['X_train']
    y_train = cfg_dict['y_train']

    X_test = cfg_dict['X_test']
    y_test = cfg_dict['y_test']


    if X_test.empty:
        logging.warning(f"No test data for {config_key}")
        return None

    idx_out = y_test.index

    if embedding == "baseline":
        X_test = X_test[[] if gender_specific else ["gender"]]
    elif embedding.endswith("_combined"):
        base_emb = embedding.replace("_combined", "")
        X_test = X_test.filter(like=base_emb)
        baseline_cols = [] if gender_specific else ["gender"]
        # X_test = pd.concat([X_test, X_test[baseline_cols]], axis=1)
    else:
        X_test = X_test.filter(like=embedding)

    model = train_final_model(X_train, y_train, best_params,
                              model_type, is_regression,
                              is_balance, seed_num)
    if model is None:
        logging.warning(f"Model train failed for {config_key}")
        return None

    eval_ = evaluate(model, X_test, y_test, is_regression)
    if eval_ is None:
        logging.warning(f"Eval failed for {config_key}")
        return None

    pd.DataFrame({
        "index": idx_out,
        "true_values": eval_["true_values"],
        "predictions": eval_["predictions"],
    }).to_csv(os.path.join(
        condition_dir, f"test_predictions_{embedding}_{seed_num}.csv"),
        index=False)

    tr = {
        "Target": target,
        "Embedding": embedding,
        "Model Type": model_type,
        "Is Regression": is_regression,
        "Evaluation": "Test Set",
        "Best Parameters": best_params,
    }
    if gender_label:
        tr["Gender"] = gender_label

    for m, v in eval_.items():
        if m not in ("predictions", "true_values"):
            tr[m] = v

    return tr

# Check if study exists
def study_exists(study_name, storage=OPTUNA_STORAGE):
    """Check if an Optuna study with the given name already exists.
    
    Args:
        study_name: Name of the study to check
        storage: Optuna storage to check
        
    Returns:
        bool: True if study exists, False otherwise
    """
    try:
        all_studies = optuna.get_all_study_summaries(storage=storage)
        study_names = [study.study_name for study in all_studies]
        return study_name in study_names
    except Exception as e:
        logging.error(f"Error checking if study exists: {str(e)}")
        return False

# Function to create or load a study using the shared DB
def get_or_create_study(study_name, direction, seed=SEED, force_new=FORCE_NEW_STUDIES):
    if force_new and study_exists(study_name, OPTUNA_STORAGE):
        try:
            optuna.delete_study(study_name=study_name, storage=OPTUNA_STORAGE)
            logging.info(f"Deleted existing study: {study_name}")
        except Exception as e:
            # Study might not exist, which is fine
            logging.debug(f"Study {study_name} not found or could not be deleted: {e}")
    
    study = optuna.create_study(
        study_name=study_name,
        storage=OPTUNA_STORAGE,
        direction=direction,
        load_if_exists=not force_new,
        sampler=optuna.samplers.TPESampler(seed=seed)
    )
    return study

def evaluate_conditions(
        q,
        dataframe,
        target_names,
        embeddings,
        train_ids,
        test_ids,
        is_balance,
        model_type,
        n_splits=N_SPLITS,
        seed_num=SEED,
        combine_with_baseline=True,
        gender_specific=False,
        run_hpo_phase=RUN_HPO_PHASE,        # Control parameter for Phase 1
        run_cv_phase=RUN_CV_PHASE,         # Control parameter for Phase 2
        run_test_phase=RUN_TEST_PHASE,       # Control parameter for Phase 3
        load_hpo_results=LOAD_HPO_RESULTS,    # Load previously computed HPO results
        hpo_results_path=None,     # Path to load HPO results from
        force_new_studies=FORCE_NEW_STUDIES
):
    """
    Three-phase evaluation with selective phase execution:
    1. Run HPO once per target/embedding combination (optional)
    2. Cross-validate with optimized parameters (optional)
    3. Final evaluation on test set (optional)
    
    Control which phases to run using the run_*_phase parameters.
    Previously computed HPO results can be loaded if run_hpo_phase=False and load_hpo_results=True.
    """
    
    results = []
    cv_results = []
    test_results = []
    all_hpo_tasks = []
    hpo_results = {}
    
    hpo_dir = Path(OUTPUT_PATH) / "hpo_results"
    hpo_dir.mkdir(parents=True, exist_ok=True)


    if gender_specific:
        gender_values = [0, 1]  # 0 for females, 1 for males
    else:
        gender_values = [None]  # Single pass for non-gender-specific analysis

    # Initialize configuration dictionary
    hpo_configs = {}
    
    # Setup configurations even if we're not running HPO
    for gender in gender_values:
        if gender is not None:
            gender_label = "female" if gender == 0 else "male"
            current_df = dataframe.loc[dataframe["gender"] == gender].copy()
        else:
            gender_label = None
            current_df = dataframe.copy()

        for target in tqdm(target_names):
            try:
                if target not in current_df.columns:
                    logging.warning(f"Target {target} not found in current dataframe")
                    continue

            # Create train dataset (ensure we're only using train_ids)
                train_mask = current_df.index.to_series().apply(
                lambda x: x.split("_")[0] in train_ids)
                train_df = current_df.loc[train_mask].dropna(subset=[target])
                
                test_mask = current_df.index.to_series().apply(
                    lambda x: x.split("_")[0] in test_ids
                )
                test_df = current_df.loc[test_mask].dropna(subset=[target])
                
                if train_df.empty:
                    logging.warning(f"No training data for {target} ({gender_label or 'all'})")
                    continue

                if test_df.empty and run_test_phase:
                    logging.warning(f"No test data for {target} ({gender_label or 'all'})")
                    continue
                
                if target in {"month_cos", "month_sin"}:
                  # Special handling: combine both cos and sin as 2D target
                    y_train = train_df[["month_cos", "month_sin"]].values  # shape (n_samples, 2)
                    y_test = test_df[["month_cos", "month_sin"]].values
                else:
                    y_train = train_df[target].copy()
                    y_test = test_df[target].copy()

                original_train_indices = train_df.index
                original_test_indices = test_df.index

                # Determine if regression task
                unique_classes = len(np.unique(y_train))
                is_regression = unique_classes > 2
                
                for embedding in embeddings:
                    # Prepare features
                    if embedding == "baseline":
                        X_train = train_df[
                            [] if gender_specific else ["gender"]
                        ].copy()
                        X_test = test_df[
                            [] if gender_specific else ["gender"]
                        ].copy()
                    else:
                        X_train = train_df[[col for col in train_df.columns if embedding in col]].copy()
                        X_test = test_df[[col for col in test_df.columns if embedding in col]].copy()
                        if combine_with_baseline:
                            baseline_cols = ["age"] if gender_specific else ["age", "gender"]
                            X_train = pd.concat([X_train, train_df[baseline_cols]], axis=1)
                            embedding = f"{embedding}_combined"
                            X_test = pd.concat([X_test, test_df[baseline_cols]], axis=1)

                    # Create output directories
                    condition_dir = Path(OUTPUT_PATH) / "prediction_results" / f"{target}_{gender_label or 'all'}"
                    condition_dir.mkdir(parents=True, exist_ok=True)
                    
                    # Create unique config key
                    config_key = f"{target}_{gender_label or 'all'}_{embedding}_{seed_num}"
                    hpo_result_file = hpo_dir / f"{config_key}_best_params.json"
                    
                    # Store config for phases 2 and 3
                    hpo_configs[config_key] = {
                        'study_name': f"{model_type}__{target}_{embedding}_{gender_label or 'all'}_{SEED}_hpo",
                        'target': target,
                        'embedding': embedding,
                        'gender_label': gender_label,
                        'is_regression': is_regression,
                        'X_train': X_train,
                        'y_train': y_train,
                        'X_test': X_test,
                        'y_test': y_test,
                        'original_train_indices': original_train_indices,
                        'original_test_indices': original_test_indices,
                        'condition_dir': condition_dir,
                        'hpo_result_file': hpo_result_file
                    }
                    # Extract group IDs from record indices
                    group_ids = []
                    for idx in X_train.index:
                        # Assuming format like "123_456", extract the ID part before first underscore
                        group_id = idx.split('_')[0]
                        group_ids.append(group_id)

                    # Submit HPO job (one per config) if enabled
                    if run_hpo_phase:
                        logging.info(f"Submitting HPO task for {config_key}")
                        try:
                            hpo_ticket = q.method(
                                run_hpo,
                                (
                                    X_train,
                                    y_train,
                                    model_type,
                                    is_balance,
                                    group_ids,
                                    N_SPLITS,
                                    NUM_TRIALS,
                                    seed_num,
                                    min(NUM_THREAD, 4),
                                    hpo_configs[config_key]['study_name'],
                                    force_new_studies
                                ),
                            )
                            
                            all_hpo_tasks.append({
                                'ticket': hpo_ticket,
                                'config_key': config_key,
                                'hpo_result_file': hpo_result_file,
                                'study_name': hpo_configs[config_key]['study_name']
                            })
                            
                        except Exception as e:
                            logging.error(f"Error submitting HPO task for {config_key}: {str(e)}")
                            continue
            
            except ValueError as e:
                logging.error(f"Skipping {target} for {gender_label or 'all'}: {str(e)}")
                continue
    
    # PHASE 1: Hyperparameter Optimization
    if run_hpo_phase:
        logging.info("PHASE 1: Running hyperparameter optimization")
        
        # Wait for HPO tasks to complete
        for task_info in tqdm(all_hpo_tasks, total=len(all_hpo_tasks), desc="Waiting for HPO tasks"):
            try:
                hpo_result = q.waitforresult(task_info['ticket'], _assert_on_errors=False)
                config_key = task_info['config_key']
                
                if hpo_result is None:
                    logging.warning(f"No HPO result for {config_key}")
                    continue
                
                # Save HPO results to file
                with open(task_info['hpo_result_file'], 'w') as f:
                    json.dump(hpo_result, f, indent=2)
                
                # Get the best parameters
                best_params = hpo_result.get('best_params', {})
                if not best_params:
                    logging.warning(f"No valid best_params for {config_key}")
                    continue
                
                hpo_results[config_key] = best_params
                logging.info(f"HPO completed for {config_key} with value: {hpo_result.get('best_value', 'N/A')}")
                
            except Exception as e:
                logging.error(f"Error processing HPO for {task_info['config_key']}: {str(e)}")
                continue
   
    elif load_hpo_results:
        logging.info("Loading previously computed HPO results")

        def load_best_params_from_file(file, config_key):
            try:
                with file.open("r") as f:
                    hpo_result = json.load(f)
                    best_params = hpo_result.get("best_params", {})
                    if best_params:
                        hpo_results[config_key] = best_params
                        logging.info(f"Loaded HPO results for {config_key}")
            except Exception as e:
                logging.error(f"Error loading HPO results from {file.name}: {e}")

        if hpo_results_path:
            hpo_dir = Path(hpo_results_path)
            if hpo_dir.is_dir():
                for file in hpo_dir.glob("*_best_params.json"):
                    config_key = file.stem.replace("_best_params", "")
                    if config_key in hpo_configs:
                        load_best_params_from_file(file, config_key)
                logging.info(f"Loaded {len(hpo_results)} HPO results for {len(hpo_configs)} configurations")
                if len(hpo_results) != len(hpo_configs):    
                    logging.warning(f"HPO results: {len(hpo_results)} and HPO configs: {len(hpo_configs)}")
        else:
            for config_key, config in hpo_configs.items():
                file_path = Path(config['hpo_result_file'])
                if file_path.exists():
                    load_best_params_from_file(file_path, config_key)

    all_studies = optuna.get_all_study_summaries(storage=OPTUNA_STORAGE)
    logging.info(f"Total number of studies in the database: {len(all_studies)}")

    # PHASE 2: Cross-validation with optimized parameters
    if run_cv_phase:
        if not hpo_results and (not load_hpo_results or not run_hpo_phase):
            logging.warning("No HPO results available for CV phase …")
        else:
            logging.info("PHASE 2: Running cross‑validation with optimized parameters")

            cv_tickets = []
        
            for config_key, best_params in hpo_results.items():
                if config_key not in hpo_configs:
                    logging.warning(f"Missing config for {config_key}, skipping CV")
                    continue

                # submit each configuration to the queue
                cv_tickets.append(
                    q.method(
                        run_cv_for_config,
                        (config_key,
                        hpo_configs[config_key],
                        best_params,
                        model_type,
                        N_SPLITS,
                        seed_num,
                        is_balance)
                    )
                )

            # gather results
            for ticket in cv_tickets:
                try:
                    avg_res = q.waitforresult(ticket, _assert_on_errors=False)
                    if avg_res:
                        cv_results.append(avg_res)
                except Exception as e:
                    logging.error(f"CV worker error: {e}")
        
    # PHASE 3: Final evaluation on test set
    if run_test_phase:
        if not hpo_results and (not load_hpo_results or not run_hpo_phase):
            logging.warning("No HPO results available …")
        else:
            logging.info("PHASE 3: Final evaluation on test set")

            test_tickets = []
            for cfg_key, best_params in hpo_results.items():
                if cfg_key not in hpo_configs:
                    logging.warning(f"Missing config for {cfg_key}, skipping test")
                    continue

                test_tickets.append(
                    q.method(
                        run_test_for_config,
                        (cfg_key,
                        hpo_configs[cfg_key],
                        best_params,
                        model_type,
                        gender_specific,
                        is_balance,
                        test_ids,
                        dataframe,
                        seed_num)
                    )
                )

            # collect results
            for t in test_tickets:
                try:
                    res = q.waitforresult(t, _assert_on_errors=False)
                    if res:
                        test_results.append(res)
                except Exception as e:
                    logging.error(f"Test worker error: {e}")

    # Combine all results
    results = cv_results + test_results
    results_df = pd.DataFrame(results)
    
    # Save separate result files for CV and test
    if run_cv_phase and cv_results:
        cv_df = pd.DataFrame(cv_results)
        if not cv_df.empty:
            cv_path = os.path.join(
                OUTPUT_PATH,
                "prediction_results",
                f"cv_results_seed_{seed_num}_{model_type}.csv"
            )
            cv_df.to_csv(cv_path, index=False)
    
    if run_test_phase and test_results:
        test_df = pd.DataFrame(test_results)
        if not test_df.empty:
            test_path = os.path.join(
                OUTPUT_PATH,
                "prediction_results",
                f"test_results_seed_{seed_num}_{model_type}.csv"
            )
            test_df.to_csv(test_path, index=False)
    
    return results_df

def main(gender_specific=GENDER_SPECIFIC, debug_mode=DEBUG_MODE, force_new_studies=FORCE_NEW_STUDIES):
    """Main function to run the voice analysis pipeline with cross-validation.
    
    Args:
        gender_specific (bool): Whether to run gender-specific analysis
        debug_mode (bool): If True, runs minimal test with only MFCC embeddings and two conditions
    """
    logging.info("Starting voice analysis pipeline")
    logging.info(f"Configuration: gender_specific={gender_specific}, debug_mode={debug_mode}")
    logging.info(f"Log file: {LOG_FILE}")
    
    # Calculate expected workload for better planning
    if debug_mode:
        num_embeddings = len(DEBUG_EMBEDDINGS)
        num_targets = len(DEBUG_TARGETS)
    else:
        num_embeddings = len(EMBEDDING_CONFIGS) + 1  # +1 for baseline
        num_targets = len(CURATED_CONDITIONS) + len(TARGET_NAMES_CALCULATED)
    
    num_genders = 2 if gender_specific else 1
    expected_configs = num_embeddings * num_targets * num_genders
    expected_trials = expected_configs * NUM_TRIALS

    logging.info("=== EXECUTION PLAN ===")
    logging.info(f"Embeddings: {num_embeddings} {'(DEBUG mode)' if debug_mode else ''}")
    logging.info(f"Target conditions: {num_targets} {'(DEBUG mode)' if debug_mode else ''}")
    logging.info(f"Gender-specific: {gender_specific} ({num_genders} gender groups)")
    logging.info(f"Expected configurations: {expected_configs}")
    logging.info(f'Expected Pipeline Steps: \n\tHyperparameter Optimization: {RUN_HPO_PHASE}, \n\tCross-Validation: {RUN_CV_PHASE}, \n\tand Test Set Evaluation: {RUN_TEST_PHASE}')
    if RUN_HPO_PHASE:
        logging.info(f"Expected HPO trials: {expected_trials} ({NUM_TRIALS} per config)")
    if RUN_CV_PHASE:
        logging.info(f"Expected CV folds:{expected_configs}*{N_SPLITS} = {expected_configs*N_SPLITS} folds")
    if RUN_TEST_PHASE:
        logging.info(f"Expected Test Set Evaluation: {RUN_TEST_PHASE} ({expected_configs} configs)")
    logging.info("=====================")
    
    # Track actual execution metrics
    actual_configs = 0
    actual_trials = 0
    actual_files = 0
    
    # Load subject details
    logging.info("Loading subject details...")
    subject_details_table = pd.read_csv(Path(SUBJECT_DETAILS_TABLE), index_col="filename")

    train_ids = pd.read_csv(Path(DATA_DIR) / "all_ids.txt", header=None).values.flatten().tolist()
    train_ids = [str(id) for id in train_ids]

    test_ids = pd.read_csv(Path(RealDeepFolderPath) / "test_ids.txt", header=None).values.flatten().tolist()
    test_ids = [str(id) for id in test_ids]
    logging.info(f"Train IDs: {len(train_ids)}, Test IDs: {len(test_ids)}")

    # if cfg.get("USE_FULL_DATA_FOR_CV", False):
    #     logging.info("USE_FULL_DATA_FOR_CV = True → using every recording inside cross-validation")
    #     train_ids = pd.read_csv(Path(DATA_DIR) / "all_ids.txt", header=None).values.flatten().tolist()  
    #     train_ids = [str(id) for id in train_ids]         # merge the two lists
    #     test_ids  = []                  # empty list signals ‘no test phase’
    #     #global RUN_TEST_PHASE           # so evaluate_conditions sees the update
    #     #RUN_TEST_PHASE = False
    #     logging.info(f"All IDs: {len(train_ids)}")


    # Create dataset
    mask = subject_details_table.index.to_series().apply(
        lambda x: x.split("_")[0] in train_ids+test_ids
    )
    dataset = subject_details_table[mask]

    # Load embeddings
    def load_embeddings_subset(requested_embeddings, ids_to_keep, cfg):
        """Return {etype: DataFrame} selecting only rows whose recording id ∈ ids_to_keep."""
        result = {}
        for etype in requested_embeddings:
            parquet = MEAN_DIR / f"{etype}.parquet"
            if not parquet.exists():
                raise FileNotFoundError(f"Pre-aggregated file {parquet} missing. "
                                        f"Run precompute_mean_embeddings.py.")

            df = pd.read_parquet(parquet)           # milliseconds
            # filter by id whitelist + any other experiment-time criteria
            mask = df.index.str.split("_").str[0].isin(ids_to_keep)
            result[etype] = df.loc[mask]
            logging.info(f"Loaded {len(result[etype])} rows for {etype} from available {len(df)} rows")
        return result

    # Load embeddings
    if debug_mode:
        logging.info(f"Running in debug mode - using {len(DEBUG_EMBEDDINGS)} embeddings and {len(DEBUG_TARGETS)} conditions")
        embeddings_to_load = DEBUG_EMBEDDINGS
    else:
        embeddings_to_load = list(EMBEDDING_CONFIGS.keys())

    embeddings = load_embeddings_subset(embeddings_to_load, train_ids + test_ids, cfg)
    
    combined_embeddings = pd.concat(list(embeddings.values()), axis=1)
    # Remove duplicate columns by keeping the first occurrence of each column name.
    combined_embeddings = combined_embeddings.loc[:, ~combined_embeddings.columns.duplicated(keep='first')]
    combined_embeddings["subject_id"] = combined_embeddings.index.str.split("_").str[0]
    combined_embeddings["recording"] = combined_embeddings.index
    dataset["recording"] = dataset.index.str.replace(".flac", "", regex=False)
    embeddings_and_clinical_baseline = combined_embeddings.merge(
        dataset, on="recording", how="left"
    )
    embeddings_and_clinical_baseline.set_index("recording", inplace=True)
    # Define targets
    if debug_mode:
        all_targets = DEBUG_TARGETS
    else:
        all_targets = CURATED_CONDITIONS + TARGET_NAMES_CALCULATED

    # Create necessary directories
    logging.info("Creating output directories...")
    base = Path(OUTPUT_PATH)
    (base / "prediction_results" / "plots").mkdir(parents=True, exist_ok=True)
    (base / "data" / "predictions").mkdir(parents=True, exist_ok=True)
    (base / "data" / "metrics_results").mkdir(parents=True, exist_ok=True)

    # Setup distributed computing
    logging.info("Setting up distributed computing...")
    sethandlers()
    os.chdir("/net/mraid20/export/genie/LabData/Analyses/DeepVoiceFolder/qp")
    #qp = fakeqp
    with qp(**QUEUE_PARAMS) as q:
        q.startpermanentrun()

        # Run evaluation with cross-validation
        logging.info("Starting evaluation...")
        results_df = evaluate_conditions(
            q=q,
            dataframe=embeddings_and_clinical_baseline,
            target_names=all_targets,
            embeddings=embeddings_to_load,
            train_ids=train_ids,
            test_ids=test_ids,
            is_balance=BALANCE_CLASSES,
            model_type=MODEL_TYPE,
            n_splits=N_SPLITS,
            seed_num=SEED,
            combine_with_baseline=COMBINE_WITH_BASELINE,
            gender_specific=gender_specific,
            run_hpo_phase=RUN_HPO_PHASE,
            run_cv_phase=RUN_CV_PHASE,
            run_test_phase=RUN_TEST_PHASE,
            load_hpo_results=LOAD_HPO_RESULTS,
            hpo_results_path=Path(OUTPUT_PATH) / "hpo_results",
            force_new_studies=force_new_studies
        )

        # Save full results
        results_path = os.path.join(
            OUTPUT_PATH,
            "prediction_results",
            f"cv_results_seed_{SEED}_{MODEL_TYPE}_HPO_all_embeddings{'_gender_specific' if gender_specific else ''}{'_debug' if debug_mode else ''}.csv"
        )
        logging.info(f"Saving results to {results_path}")
        results_df.to_csv(results_path, index=False)

        # Count actual number of configurations and files
        for root, dirs, files in os.walk(os.path.join(OUTPUT_PATH, "hpo_results")):
            for file in files:
                if file.endswith("_best_params.json"):
                    actual_files += 1
                    
        # Count actual trials from Optuna database
        all_studies = optuna.get_all_study_summaries(storage=OPTUNA_STORAGE)
        for study in all_studies:
            if hasattr(study, 'n_trials'):
                actual_trials += study.n_trials
        
        
        # Log actual execution statistics
        logging.info("=== EXECUTION SUMMARY ===")
        logging.info(f"Planned configurations: {expected_configs}, Actual: {actual_files}")
        logging.info(f"Planned HPO trials: {expected_trials}, Actual: {actual_trials}")
        logging.info(f"Planned result files: {expected_configs}, Actual: {actual_files}")
        
        # Calculate completion percentages
        config_completion = (actual_configs / expected_configs) * 100 if expected_configs > 0 else 0
        trial_completion = (actual_trials / expected_trials) * 100 if expected_trials > 0 else 0
        file_completion = (actual_files / expected_configs) * 100 if expected_configs > 0 else 0
        
        logging.info(f"Configuration completion: {config_completion:.1f}%")
        logging.info(f"Trial completion: {trial_completion:.1f}%")
        logging.info(f"Result file completion: {file_completion:.1f}%")
        logging.info("========================")

        if results_df.empty:
            logging.warning("No results to save")
        else:
            # Create comprehensive summary statistics
            logging.info("Creating summary statistics...")
            summary_stats = []
            groupby_cols = ['Target', 'Embedding', 'Gender'] if gender_specific else ['Target', 'Embedding']

            for group_key, group in results_df.groupby(groupby_cols):
                stats = dict(zip(groupby_cols, group_key))
                stats.update({
                    'Model Type': group['Model Type'].iloc[0],
                    'Is Regression': group['Is Regression'].iloc[0],
                    'Number of Folds': len(group),
                })

                # Store fold-level metrics
                if stats['Is Regression']:
                    possible_metrics = ['mse', 'r2', 'rmse', 'mae']
                else:
                    possible_metrics = ['auc', 'f1', 'auprc', 'accuracy', 'prevalence']

                # Find available metrics in the group's columns
                metrics = [metric for metric in possible_metrics if any(col.startswith(metric) for col in group.columns)]

                # Store all fold results and compute statistics
                for metric in metrics:
                    # Check for both direct metric and mean versions
                    if metric in group.columns:
                        fold_values = group[metric].tolist()
                        stats[f'{metric}_per_fold'] = fold_values
                        stats[f'{metric}_mean'] = np.mean(fold_values)
                        stats[f'{metric}_std'] = np.std(fold_values)
                        stats[f'{metric}_min'] = np.min(fold_values)
                        stats[f'{metric}_max'] = np.max(fold_values)
                    elif f'{metric}_mean' in group.columns:
                        # If we only have the mean values
                        stats[f'{metric}_mean'] = group[f'{metric}_mean'].iloc[0]
                        if f'{metric}_std' in group.columns:
                            stats[f'{metric}_std'] = group[f'{metric}_std'].iloc[0]

                        # Try to get min/max if available
                        if f'{metric}_min' in group.columns:
                            stats[f'{metric}_min'] = group[f'{metric}_min'].iloc[0]
                        if f'{metric}_max' in group.columns:
                            stats[f'{metric}_max'] = group[f'{metric}_max'].iloc[0]

                # Store best parameters from each fold
                if 'Best Parameters Per Fold' in group.columns:
                    stats['Best Parameters Per Fold'] = group['Best Parameters Per Fold'].tolist()
                elif 'best_params' in group.columns:
                    stats['Best Parameters Per Fold'] = group['best_params'].tolist()
                else:
                    # If neither column exists, store empty list
                    stats['Best Parameters Per Fold'] = []

                summary_stats.append(stats)

            # Save summary statistics
            summary_df = pd.DataFrame(summary_stats)
            summary_path = os.path.join(
                OUTPUT_PATH,
                "prediction_results",
                f"cv_summary_seed_{SEED}_{MODEL_TYPE}_HPO_first_seg_CV_rest_baseline{'_gender_specific' if gender_specific else ''}{'_debug' if debug_mode else ''}.csv"
            )
            logging.info(f"Saving summary statistics to {summary_path}")
            summary_df.to_csv(summary_path, index=False)
            logging.info("Pipeline completed successfully")
        

if __name__ == "__main__":
    try:
        start = datetime.now()
        logging.info("Starting script execution")
        main(gender_specific=GENDER_SPECIFIC, debug_mode=DEBUG_MODE, force_new_studies=FORCE_NEW_STUDIES)
    except Exception as e:
        logging.exception("Unhandled exception occurred!")
        raise  # Optionally re-raise if you want the script to crash
    finally:
        end = datetime.now()
        duration = end - start
        logging.info(f"Total execution time: {duration}")

