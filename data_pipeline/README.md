# Data pipeline

```bash
# Create/update feature store
make feast_apply

# Build
make build_image && make deploy_dags

# Go to airflow UI
# Set variable CODE_DIR=path/to/credit-ranking-code
# Run dags

# Deploy feature repo to training pipeline
make deploy_feature_repo
```
