import pandas as pd
from feast import FeatureStore

store = FeatureStore(repo_path="../feature_repo")

features = store.get_online_features(
    features=[
        'credit_stats:income_expenditure_difference', 
        "credit_stats:income", 
        "credit_stats:total_expenses", 
        'credit_stats:loan_term', 
        'credit_stats:expected_loan_interest'
    ],
    entity_rows=[{"id": 0}, {"id": 100}],
).to_dict()


def print_online_features(features):
    for key, value in sorted(features.items()):
        print(key, " : ", value)


print_online_features(features)
