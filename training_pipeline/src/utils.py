import json
import logging
import os
import sys
from pathlib import Path

import pandas as pd
import numpy as np
from dotenv import load_dotenv

load_dotenv()


class AppConst:
    LOG_LEVEL = logging.DEBUG
    DATA_EXTRACTION = "data_extraction"
    DATA_VALIDATION = "data_validation"
    DATA_PREPARATION = "data_preparation"
    MODEL_TRAINING = "model_training"
    MODEL_EVALUATION = "model_evaluation"
    MODEL_VALIDATION = "model_validation"
    MLFLOW_MODEL_PATH_PREFIX = "model"


class AppPath:
    # set TRAINING_PIPELINE_DIR in dev environment for quickly testing the code
    ROOT = Path(os.environ.get("TRAINING_PIPELINE_DIR", "/training_pipeline"))
    DATA = ROOT / "data"
    DATA_SOURCES = ROOT / "data_source"
    FEATURE_REPO = ROOT / "feature_repo"
    ARTIFACTS = ROOT / "artifacts"

    TRAINING_PQ = ARTIFACTS / "training.parquet"
    TRAIN_X_PQ = ARTIFACTS / "train_x.parquet"
    TRAIN_Y_PQ = ARTIFACTS / "train_y.parquet"
    TEST_X_PQ = ARTIFACTS / "test_x.parquet"
    TEST_Y_PQ = ARTIFACTS / "test_y.parquet"
    RUN_INFO = ARTIFACTS / "run_info.json"
    EVALUATION_RESULT = ARTIFACTS / "evaluation.json"
    REGISTERED_MODEL_VERSION = ARTIFACTS / "registered_model_version.json"

    def __init__(self) -> None:
        AppPath.ARTIFACTS.mkdir(parents=True, exist_ok=True)


class Config:
    def __init__(self) -> None:

        self.random_seed = int(os.environ.get("RANDOM_SEED"))
        self.continuous_features = {
            "income_expenditure_difference": np.float64,
            "income": np.float64,
            "total_expenses": np.float64,
            "total_income": np.float64,
            "loan_term": np.float64,
            "salary_allowance": np.float64,
            "number_of_products_in_use": np.int64,
            "casa_balance": np.float64,
            "number_of_dependents": np.int64,
            "age": np.int64,
            "term_deposit_balance": np.float64,
            "number_of_non_credit_products": np.int64,
            "work_tenure": np.float64,
            "bank_debt_balance": np.float64,
            "number_of_banks_with_outstanding_debt": np.int64,
            "loan_amount": np.float64,
            "expected_loan_interest": np.float64,
            "duration_of_relationship_with_the_bank": np.int64,
            "proposed_term": np.float64,
        }
        self.category_features = {
            "working_agency": np.object_,
            "type_of_residence": np.object_,
            "bank_product": np.object_,
            "customer_segment": np.object_,
            "marital_status": np.object_,
            "educational_level": np.object_,
            "insurance": np.object_,
            "position": np.object_,
            "housing": np.object_,
            "debt_repayment_source": np.object_,
            "labor_contract": np.object_,
            "economic_sector": np.object_,
            "debt_group_information": np.object_,
            "overdue_history": np.object_,
        }
        self.feature_dict = {"result": np.object_}
        self.feature_dict.update(self.continuous_features)
        self.feature_dict.update(self.category_features)
        self.target_col = os.environ.get("TARGET_COL")
        self.labels = ['A', 'A+', 'A-', 'AA', 'AA+', 'AA-', 'AAA', 'B', 'BB', 'BBB']
        self.test_size = float(os.environ.get("TEST_SIZE"))
        self.experiment_name = os.environ.get("EXPERIMENT_NAME")
        self.mlflow_tracking_uri = os.environ.get("MLFLOW_TRACKING_URI")
        self.max_depth = int(os.environ.get("MAX_DEPTH"))
        self.n_estimators = int(os.environ.get("N_ESTIMATORS"))
        self.learning_rate = float(os.environ.get("LEARNING_REATE"))
        self.subsample = float(os.environ.get("SUBSAMPLE"))
        self.accuracy_threshold = float(os.environ.get("ACCURACY_THRESHOLD"))
        self.precision_threshold = float(os.environ.get("PRECISION_THRESHOLD"))
        self.recall_threshold = float(os.environ.get("RECALL_THRESHOLD"))
        self.f1_threshold = float(os.environ.get("F1_THRESHOLD"))
        self.registered_model_name = os.environ.get("REGISTERED_MODEL_NAME")


class RunInfo:
    def __init__(self, run_id) -> None:
        self.path = AppPath.RUN_INFO
        self.run_id = run_id

    def save(self):
        run_info = {
            "run_id": self.run_id,
        }
        dump_json(run_info, self.path)

    @staticmethod
    def load(path):
        data = load_json(path)
        run_info = RunInfo(data["run_id"])
        return run_info


class EvaluationResult:
    def __init__(self, accuracy, precision, recall, f1) -> None:
        self.path = AppPath.EVALUATION_RESULT
        self.accuracy = accuracy
        self.precision = precision
        self.recall = recall
        self.f1 = f1

    def __str__(self) -> str:
        return f"Accuracy: {self.accuracy}, Precision: {self.precision}, Recall: {self.recall}, F1: {self.f1}"

    def save(self):
        eval_result = {
            "accuracy": self.accuracy,
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1
        }
        dump_json(eval_result, self.path)

    @staticmethod
    def load(path):
        data = load_json(path)
        eval_result = EvaluationResult(
            data["accuracy"],
            data["precision"],
            data["recall"],
            data["f1"]
        )
        return eval_result


class Log:
    log: logging.Logger = None

    def __init__(self, name="") -> None:
        if Log.log == None:
            Log.log = self._init_logger(name)

    def _init_logger(self, name):
        logger = logging.getLogger(name)
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)
        logger.setLevel(AppConst.LOG_LEVEL)
        return logger


def inspect_dir(path):
    Log().log.info(f"inspect_dir {path}")
    path = Path(path)
    if not path.exists():
        Log().log.info(f"Path {path} doesn't exist")
        return
    elif path.is_file():
        Log().log.info(f"Path {path} is file")
        return

    paths = os.listdir(path)
    paths = sorted(paths)
    for path in paths:
        Log().log.info(path)


def inspect_curr_dir():
    cwd = os.getcwd()
    Log().log.info(f"current dir: {cwd}")
    inspect_dir(cwd)


def load_df(path) -> pd.DataFrame:
    Log().log.info(f"start load_df {path}")
    df = pd.read_parquet(path, engine="fastparquet")
    return df


def to_parquet(df: pd.DataFrame, path):
    Log().log.info(f"start to_parquet {path}")
    df.to_parquet(path, engine="fastparquet")


def dump_json(dict_obj: dict, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(dict_obj, f)


def load_json(path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data
