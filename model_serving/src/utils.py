import json
import logging
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv

load_dotenv()


class AppConst:
    LOG_LEVEL = logging.DEBUG
    BENTOML_MODEL_SAVING = "bentoml_model_saving"
    BENTOML_SERVICE = "bentoml_service"
    DATA_EXTRACTION = "data_extraction"
    BATCH_PREDICTION = "batch_prediction"


class AppPath:
    # set MODEL_SERVING_DIR in dev environment for quickly testing the code
    ROOT = Path(os.environ.get("MODEL_SERVING_DIR", "/model_serving"))
    DATA = ROOT / "data"
    DATA_SOURCES = ROOT / "data_sources"
    FEATURE_REPO = ROOT / "feature_repo"
    ARTIFACTS = ROOT / "artifacts"

    BATCH_INPUT_PQ = ARTIFACTS / "batch_input.parquet"
    BATCH_OUTPUT_PQ = ARTIFACTS / "batch_output.parquet"

    def __init__(self) -> None:
        AppPath.ARTIFACTS.mkdir(parents=True, exist_ok=True)


class Config:
    def __init__(self) -> None:
        import numpy as np

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
        self.mlflow_tracking_uri = os.environ.get("MLFLOW_TRACKING_URI")
        self.batch_input_file = os.environ.get("BATCH_INPUT_FILE")
        self.registered_model_file = os.environ.get("REGISTERED_MODEL_FILE")
        self.monitoring_service_api = os.environ.get("MONITORING_SERVICE_API")


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


# the encoder helps to convert NumPy types in source data to JSON-compatible types
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.void):
            return None

        if isinstance(obj, (np.generic, np.bool_)):
            return obj.item()

        if isinstance(obj, np.ndarray):
            return obj.tolist()

        return obj


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
