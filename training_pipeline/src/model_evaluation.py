import mlflow
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from utils import *

Log(AppConst.MODEL_EVALUATION)
AppPath()


def eval_metrics(actual, pred):
    accuracy = accuracy_score(actual, pred)
    precision = precision_score(actual, pred, average='macro')
    recall = recall_score(actual, pred, average='macro')
    f1 = f1_score(actual, pred, average='macro')
    return accuracy, precision, recall, f1


def evaluate_model():
    Log().log.info("start evaluate_model")
    inspect_curr_dir()

    run_info = RunInfo.load(AppPath.RUN_INFO)
    Log().log.info(f"loaded run_info {run_info.__dict__}")

    config = Config()
    Log().log.info(f"config: {config.__dict__}")
    mlflow.set_tracking_uri(config.mlflow_tracking_uri)

    model = mlflow.pyfunc.load_model(
        f"runs:/{run_info.run_id}/{AppConst.MLFLOW_MODEL_PATH_PREFIX}"
    )
    Log().log.info(f"loaded model {model.__dict__}")


    config = Config()
    test_x = load_df(AppPath.TEST_X_PQ)
    test_y = load_df(AppPath.TEST_Y_PQ)
    label_to_ids = {l: i for i, l in enumerate(config.labels)}
    test_y = test_y.replace(label_to_ids)

    predicted_qualities = model.predict(test_x)
    accuracy, precision, recall, f1 = eval_metrics(test_y, predicted_qualities)

    # Write evaluation result to file
    eval_result = EvaluationResult(accuracy, precision, recall, f1)
    Log().log.info(f"eval result: {eval_result}")
    eval_result.save()
    inspect_dir(eval_result.path)


if __name__ == "__main__":
    evaluate_model()
