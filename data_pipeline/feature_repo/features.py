from datetime import timedelta

from feast import FeatureView, Field
# from feast.stream_feature_view import stream_feature_view
from feast.types import Float32, Int32, String
from pyspark.sql import DataFrame

from data_sources import credit_stats_batch_source
from entities import credit

credit_stats_view = FeatureView(
    name="credit_stats",
    description="credit features",
    entities=[credit],
    ttl=timedelta(days=36500),
    schema=[
        Field(name="id", dtype=Int32),
        Field(name="income_expenditure_difference", dtype=Float32),
        Field(name="income", dtype=Float32),
        Field(name="working_agency", dtype=String),
        Field(name="total_expenses", dtype=Float32),
        Field(name="type_of_residence", dtype=String),
        Field(name="bank_product", dtype=String),
        Field(name="total_income", dtype=Float32),
        Field(name="loan_term", dtype=Float32),
        Field(name="salary_allowance", dtype=Float32),
        Field(name="number_of_products_in_use", dtype=Float32),
        Field(name="casa_balance", dtype=Float32),
        Field(name="customer_segment", dtype=String),
        Field(name="marital_status", dtype=String),
        Field(name="number_of_dependents", dtype=Float32),
        Field(name="age", dtype=Float32),
        Field(name="term_deposit_balance", dtype=Float32),
        Field(name="educational_level", dtype=String),
        Field(name="insurance", dtype=String),
        Field(name="position", dtype=String),
        Field(name="number_of_non_credit_products", dtype=Float32),
        Field(name="housing", dtype=String),
        Field(name="debt_repayment_source", dtype=String),
        Field(name="work_tenure", dtype=Float32),
        Field(name="bank_debt_balance", dtype=Float32),
        Field(name="number_of_banks_with_outstanding_debt", dtype=Float32),
        Field(name="loan_amount", dtype=Float32),
        Field(name="expected_loan_interest", dtype=Float32),
        Field(name="labor_contract", dtype=String),
        Field(name="duration_of_relationship_with_the_bank", dtype=Float32),
        Field(name="proposed_term", dtype=Float32),
        Field(name="economic_sector", dtype=String),
        Field(name="debt_group_information", dtype=String),
        Field(name="overdue_history", dtype=String),
        Field(name="result", dtype=String),
    ],
    online=True,
    source=credit_stats_batch_source,
    tags={},
    owner="quyanh",
)