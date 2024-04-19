import uuid

import mlflow
from mlflow.models.signature import infer_signature
from mlflow.tracking import MlflowClient
from xgboost import XGBClassifier

from sklearn.preprocessing import (
    StandardScaler,
    OneHotEncoder,
    OrdinalEncoder,
    LabelEncoder
)
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from utils import *

Log(AppConst.MODEL_TRAINING)
AppPath()


def yield_artifacts(run_id, path=None):
    """Yield all artifacts in the specified run"""
    client = MlflowClient()
    for item in client.list_artifacts(run_id, path):
        if item.is_dir:
            yield from yield_artifacts(run_id, item.path)
        else:
            yield item.path


def fetch_logged_data(run_id):
    """Fetch params, metrics, tags, and artifacts in the specified run"""
    client = MlflowClient()
    data = client.get_run(run_id).data
    # Exclude system tags: https://www.mlflow.org/docs/latest/tracking.html#system-tags
    tags = {k: v for k, v in data.tags.items() if not k.startswith("mlflow.")}
    artifacts = list(yield_artifacts(run_id))
    return {
        "params": data.params,
        "metrics": data.metrics,
        "tags": tags,
        "artifacts": artifacts,
    }


def train_model():
    Log().log.info("start train_model")
    inspect_curr_dir()

    # Setup tracking server
    config = Config()
    Log().log.info(f"config: {config.__dict__}")
    mlflow.set_tracking_uri(config.mlflow_tracking_uri)
    mlflow.set_experiment(config.experiment_name)
    Log().log.info((mlflow.get_tracking_uri(), mlflow.get_artifact_uri()))
    mlflow.sklearn.autolog()

    # Load data
    train_x = load_df(AppPath.TRAIN_X_PQ)
    train_y = load_df(AppPath.TRAIN_Y_PQ)
    label_to_ids = {l: i for i, l in enumerate(config.labels)}
    train_y = train_y.replace(label_to_ids)

    # Training
    continuous_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="mean")),
            ("scaler", StandardScaler()),
        ]
    )

    category_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("ord", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", continuous_transformer, list(config.continuous_features.keys())),
            ("category", category_transformer, list(config.category_features.keys()))
        ], remainder="passthrough"
    )
    model = Pipeline(
        [
            ("preprocessor", preprocessor),
            (
                "model",
                XGBClassifier(
                    max_depth=config.max_depth,
                    n_estimators=config.n_estimators,
                    learning_rate=config.learning_rate,
                    random_state=config.random_seed,
                    subsample=config.subsample,
                ),
            ),
        ]
    )
    print(type(model))
    model.fit(train_x, train_y)

    # Log metadata
    mlflow.set_tag("mlflow.runName", str(uuid.uuid1())[:8])
    signature = infer_signature(train_x, model.predict(train_x))
    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path=AppConst.MLFLOW_MODEL_PATH_PREFIX,
        signature=signature,
    )
    mlflow.end_run()

    # Inspect metadata
    run_id = mlflow.last_active_run().info.run_id
    Log().log.info("Logged data and model in run {}".format(run_id))
    for key, data in fetch_logged_data(run_id).items():
        Log().log.info("\n---------- logged {} ----------".format(key))
        Log().log.info(data)

    # Write latest run_id to file
    run_info = RunInfo(run_id)
    run_info.save()
    inspect_dir(run_info.path)


if __name__ == "__main__":
    train_model()
