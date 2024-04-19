from feast import FeatureStore
import pandas as pd
from datetime import datetime

store = FeatureStore(repo_path="../feature_repo")

entity_df = pd.read_parquet("../data_source/credit-dataset.parquet").head()
print(entity_df)
entity_df = entity_df[["id", "event_timestamp"]]

training_df = store.get_historical_features(
    entity_df=entity_df,
    features=[
        'credit_stats:income_expenditure_difference', 
        "credit_stats:income", 
        "credit_stats:total_expenses", 
        'credit_stats:loan_term', 
        'credit_stats:expected_loan_interest'
    ]
).to_df()
print(training_df.head())
