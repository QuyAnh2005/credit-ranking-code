# Training pipeline

```bash
# Go to data pipeline and deploy feature repo
cd ../data_pipeline
make deploy_feature_repo
cd ../training_pipeline

# To test source files at local before running in Airflow
export TRAINING_PIPELINE_DIR="path/to/credit-ranking-code/training_pipeline"
cd src
python <source_file>

# Build
make build_image && make deploy_dags

# Go to airflow UI
# Set variable CODE_DIR=path/to/credit-ranking-code
# Run dags
```
