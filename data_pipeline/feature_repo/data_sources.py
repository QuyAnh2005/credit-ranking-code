from datetime import timedelta

from feast import FileSource, KafkaSource
from feast.data_format import JsonFormat, ParquetFormat

credit_stats_parquet_file = "../data_source/credit-dataset.parquet"

credit_stats_batch_source = FileSource(
    name="credit_stats",
    file_format=ParquetFormat(),
    timestamp_field="event_timestamp",
    path=credit_stats_parquet_file,
)
