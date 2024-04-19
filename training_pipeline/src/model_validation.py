import mlflow
from utils import *

Log(AppConst.MODEL_VALIDATION)
AppPath()


def validate_model():
    Log().log.info("start validate_model")
    inspect_curr_dir()

    eval_result = EvaluationResult.load(AppPath.EVALUATION_RESULT)
    Log().log.info(f"loaded eval_result {eval_result.__dict__}")

    errors = []
    config = Config()
    Log().log.info(f"config: {config.__dict__}")
    if eval_result.accuracy < config.accuracy_threshold:
        errors.append(
            f"accuracy result {eval_result.accuracy} is not more larger threshold {config.accuracy_threshold}"
        )
    if eval_result.precision < config.precision_threshold:
        errors.append(
            f"precision result {eval_result.precision} is not more larger threshold {config.precision_threshold}"
        )
    if eval_result.recall < config.recall_threshold:
        errors.append(
            f"recall result {eval_result.recall} is not more larger threshold {config.recall_threshold}"
        )
    if eval_result.f1 < config.f1_threshold:
        errors.append(
            f"f1 result {eval_result.f1} is not more larger threshold {config.f1_threshold}"
        )

    if len(errors) > 0:
        Log().log.info(f"Model validation fails, will not register model: {errors}")
        return

    Log().log.info(f"Model validation succeeds, registering model")
    run_info = RunInfo.load(AppPath.RUN_INFO)
    Log().log.info(f"loaded run_info {run_info.__dict__}")

    mlflow.set_tracking_uri(config.mlflow_tracking_uri)
    result = mlflow.register_model(
        f"runs:/{run_info.run_id}/{AppConst.MLFLOW_MODEL_PATH_PREFIX}",
        config.registered_model_name,
    )
    dump_json(result.__dict__, AppPath.REGISTERED_MODEL_VERSION)
    inspect_dir(AppPath.REGISTERED_MODEL_VERSION)


if __name__ == "__main__":
    validate_model()
