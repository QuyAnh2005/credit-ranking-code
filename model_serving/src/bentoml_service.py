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
    ids: List[int]


class InferenceResponse(BaseModel):
    prediction: List[float]
    error: Optional[str]


@svc.api(
    input=JSON(pydantic_model=InferenceRequest),
    output=JSON(pydantic_model=InferenceResponse),
)
def inference(request: InferenceRequest, ctx: bentoml.Context) -> Dict[str, Any]:
    """
    Example request: {"request_id": "uuid-1", "ids":[1, 2, 3, 4]}
    """
    Log().log.info(f"start inference")
    response = InferenceResponse()
    try:
        Log().log.info(f"request: {request}")
        ids = request.ids

        online_features = fs.get_online_features(
            entity_rows=[{"id": id} for id in ids],
            features=[f"credit_stats:{name}" for name in feature_list],
        )
        df = pd.DataFrame.from_dict(online_features.to_dict())
        Log().log.info(f"online features: {df}")

        input_features = df.drop(["id"], axis=1)
        input_features = input_features[feature_list]
        Log().log.info(f"input_features: {input_features}")

        result = predict(input_features)
        df["prediction"] = result
        best_idx = df["prediction"].argmax()
        best_id = df["id"].iloc[best_idx]
        Log().log.info(f"best_id: {best_id}")
        Log().log.info(f"df: {df}")

        response.prediction = result.tolist()
        ctx.response.status_code = 200

        # monitor
        monitor_df = df.iloc[[best_idx]]
        monitor_df = monitor_df.assign(request_id=[request.request_id])
        monitor_df = monitor_df.assign(best_id=[best_id])
        Log().log.info(f"monitor_df: {monitor_df}")
        monitor_request(monitor_df)

    except Exception as e:
        Log().log.error(f"error: {e}")
        response.error = str(e)
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
