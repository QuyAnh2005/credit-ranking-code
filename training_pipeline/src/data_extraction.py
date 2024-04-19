import feast
import pandas as pd

from utils import *

Log(AppConst.DATA_EXTRACTION)
AppPath()


def extract_data():
    Log().log.info("start extract_data")
    inspect_curr_dir()

    # Connect to your feature store provider
    inspect_dir(AppPath.DATA_SOURCES)
    inspect_dir(AppPath.FEATURE_REPO)
    fs = feast.FeatureStore(repo_path=AppPath.FEATURE_REPO)

    # Load credit data
    inspect_dir(AppPath.DATA)
    credits = pd.read_parquet(AppPath.DATA / "credit-dataset.parquet").head(100)
    credits["event_timestamp"] = pd.to_datetime(credits["event_timestamp"])
    print(credits.columns)

    # Retrieve training data
    training_df = fs.get_historical_features(
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
            "credit_stats:result"
        ],
    ).to_df()
    print(training_df.columns)

    training_df = training_df.drop(["event_timestamp", "id"], axis=1)

    Log().log.info("----- Feature schema -----")
    Log().log.info(training_df.info())

    Log().log.info("----- Example features -----")
    Log().log.info(training_df.head())

    # Write to file
    to_parquet(training_df, AppPath.TRAINING_PQ)
    inspect_dir(AppPath.TRAINING_PQ.parent)


if __name__ == "__main__":
    extract_data()
