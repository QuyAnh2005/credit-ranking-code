from typing import Any, Dict, List, Optional

import bentoml
import feast
import mlflow
import numpy as np
import pandas as pd
import requests
from bentoml.io import JSON
from mlflow.models.signature import ModelSignature
from pydantic import BaseModel

from utils import *

Log(AppConst.BENTOML_SERVICE)
AppPath()
pd.set_option("display.max_columns", None)
config = Config()
Log().log.info(f"config: {config.__dict__}")


def save_model() -> bentoml.Model:
    Log().log.info("start save_model")
    # read from .env file registered_model_version.json, get model name, model version

    registered_model_file = AppPath.ROOT / config.registered_model_file
    Log().log.info(f"registered_model_file: {registered_model_file}")
    registered_model_dict = load_json(registered_model_file)
    Log().log.info(f"registered_model_dict: {registered_model_dict}")

    run_id = registered_model_dict["_run_id"]
    model_name = registered_model_dict["_name"]
    model_version = registered_model_dict["_version"]
    model_uri = registered_model_dict["_source"]

    mlflow.set_tracking_uri(config.mlflow_tracking_uri)
    mlflow_model = mlflow.pyfunc.load_model(model_uri=model_uri)
    Log().log.info(mlflow_model.__dict__)
    model = mlflow_model._model_impl
    model_signature: ModelSignature = mlflow_model.metadata.signature

    # construct feature list
    feature_list = []
    for name in model_signature.inputs.input_names():
        feature_list.append(name)

    # save model using bentoml
    model = mlflow.sklearn.load_model(model_uri=model_uri)
    bentoml_model = bentoml.sklearn.save_model(
        model_name,
        model,
        # model signatures for runner inference
        signatures={
            "predict": {
                "batchable": False,
            },
        },
        labels={
            "owner": "quyanh",
        },
        metadata={
            "mlflow_run_id": run_id,
            "mlflow_model_name": model_name,
            "mlflow_model_version": model_version,
        },
        custom_objects={
            "feature_list": feature_list,
        },
    )
    Log().log.info(bentoml_model.__dict__)
    return bentoml_model


bentoml_model = save_model()
feature_list = bentoml_model.custom_objects["feature_list"]
bentoml_runner = bentoml.sklearn.get(bentoml_model.tag).to_runner()
svc = bentoml.Service(bentoml_model.tag.name, runners=[bentoml_runner])
fs = feast.FeatureStore(repo_path=AppPath.FEATURE_REPO)


def predict(request: np.ndarray) -> np.ndarray:
    Log().log.info(f"start predict")
    result = bentoml_runner.predict.run(request)
    Log().log.info(f"result: {result}")
    return result


class InferenceRequest(BaseModel):
    request_id: str
    customer_id: int


class InferenceResponse(BaseModel):
    prediction: Optional[float]
    error: Optional[str]


@svc.api(
    input=JSON(pydantic_model=InferenceRequest),
    output=JSON(pydantic_model=InferenceResponse),
)
def inference(request: InferenceRequest, ctx: bentoml.Context) -> Dict[str, Any]:
    """
    Example request: {"request_id": "uuid-1", "customer_id": 0}
    """

    try:
        Log().log.info(f"start inference")

        Log().log.info(f"request: {request}")
        ids = request.customer_id
        print("-" * 100, ids)

        credits = pd.read_csv(AppPath.DATA / "batch_request.csv").head(100)
        credits = credits[credits["id"] == ids]
        credits["event_timestamp"] = pd.to_datetime(credits["event_timestamp"])

        # Retrieve training data
        df = fs.get_historical_features(
            entity_df=credits[["event_timestamp", "id"]],
            features=[
                "credit_stats:income_expenditure_difference",
                "credit_stats:income",
                "credit_stats:working_agency",
                "credit_stats:total_expenses",
                "credit_stats:type_of_residence",
                "credit_stats:bank_product",
                "credit_stats:total_income",
                "credit_stats:loan_term",
                "credit_stats:salary_allowance",
                "credit_stats:number_of_products_in_use",
                "credit_stats:casa_balance",
                "credit_stats:customer_segment",
                "credit_stats:marital_status",
                "credit_stats:number_of_dependents",
                "credit_stats:age",
                "credit_stats:term_deposit_balance",
                "credit_stats:educational_level",
                "credit_stats:insurance",
                "credit_stats:position",
                "credit_stats:number_of_non_credit_products",
                "credit_stats:housing",
                "credit_stats:debt_repayment_source",
                "credit_stats:work_tenure",
                "credit_stats:bank_debt_balance",
                "credit_stats:number_of_banks_with_outstanding_debt",
                "credit_stats:loan_amount",
                "credit_stats:expected_loan_interest",
                "credit_stats:labor_contract",
                "credit_stats:duration_of_relationship_with_the_bank",
                "credit_stats:proposed_term",
                "credit_stats:economic_sector",
                "credit_stats:debt_group_information",
                "credit_stats:overdue_history",
            ],
        ).to_df()

        Log().log.info(f"features: {df}")

        input_features = df[feature_list]
        Log().log.info(f"input_features: {input_features}")

        result = predict(input_features)
        result = result.tolist()[0]

        response = InferenceResponse.model_validate({"prediction": result, "error": ""})
        ctx.response.status_code = 200

        # monitor
        monitor_df = df.iloc[0]
        Log().log.info(f"monitor_df: {monitor_df}")
        monitor_request(monitor_df)

    except Exception as e:
        Log().log.error(f"error: {e}")
        response = InferenceResponse.model_validate({"prediction": -1, "error": str(e)})
        ctx.response.status_code = 500

    Log().log.info(f"response: {response}")
    return response


def monitor_request(df: pd.DataFrame):
    Log().log.info("start monitor_request")
    try:
        data = json.dumps(df.to_dict(), cls=NumpyEncoder)

        Log().log.info(f"sending {data}")
        response = requests.post(
            config.monitoring_service_api,
            data=data,
            headers={"content-type": "application/json"},
        )

        if response.status_code == 200:
            Log().log.info(f"Success")
        else:
            Log().log.info(
                f"Got an error code {response.status_code} for the data chunk. Reason: {response.reason}, error text: {response.text}"
            )

    except requests.exceptions.ConnectionError as error:
        Log().log.error(
            f"Cannot reach monitoring service, error: {error}, data: {data}"
        )

    except Exception as error:
        Log().log.error(f"Error: {error}")
