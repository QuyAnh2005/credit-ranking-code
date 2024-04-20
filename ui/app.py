import json
import requests

import pandas as pd
import gradio as gr

from utils import Log


ONLINE_SERVING_API = "http://localhost:8172/inference"
DATA_PATH = "../data_pipeline/data_source/credit-dataset.parquet"

with open("columns.json", "r") as data:
	columns = json.load(data)
columns = {k : v.capitalize() for k, v in columns.items()}
df = pd.read_parquet(DATA_PATH)
labels = ['A', 'A+', 'A-', 'AA', 'AA+', 'AA-', 'AAA', 'B', 'BB', 'BBB']


def send_request(request: dict) -> None:
    Log().log.info(f"start send_request")

    try:
        data = json.dumps(request)
        Log().log.info(f"sending {data}")
        response = requests.post(
            ONLINE_SERVING_API,
            data=data,
            headers={"content-type": "application/json"},
        )

        if response.status_code == 200:
            Log().log.info(f"Success.")
        else:
            Log().log.info(
                f"Status code: {response.status_code}. Reason: {response.reason}, error text: {response.text}"
            )

        return response.content

    except Exception as error:
        Log().log.info(f"Error: {error}")


def inference(customer_id):

	customer_df = df[df["id"] == customer_id]
	customer_df = customer_df.rename(columns=columns).transpose()
	customer_df = customer_df.reset_index()

	req = {
        "request_id": "Inference",
        "customer_id": customer_id,
    }
	
	response = send_request(req)
	str_data = response.decode('utf-8')
	response_dict = json.loads(str_data)
	return [labels[int(response_dict["prediction"])], customer_df.values]


title = "Xếp hạng tín dụng cá nhân"
description = "Demo"
examples = [id for id in range(10)]

gr.Interface(
    inference,
    inputs=[gr.Number(label='Customer ID')],
	outputs=[
		gr.Text(label='Xếp hạng'), 
		gr.Dataframe(headers=['	Thông tin', 'Gía trị'], label='Customer Information')
	],
	title=title,
	description=description,
	examples=examples,
).launch(debug=True)