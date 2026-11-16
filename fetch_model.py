"""Download the latest trained model from DagsHub (via MLflow) into ./model."""
import mlflow
import mlflow.sklearn

EXPERIMENT_NAME = "RandomForest_V1.0.1"
OUTPUT_DIR = "model"

# MLFLOW_TRACKING_URI, MLFLOW_TRACKING_USERNAME and MLFLOW_TRACKING_PASSWORD
# are read automatically from the environment.
exp = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
if exp is None:
    raise SystemExit(f"Experiment not found: {EXPERIMENT_NAME}")

models = mlflow.search_logged_models(
    experiment_ids=[exp.experiment_id],
    order_by=[{"field_name": "creation_time", "ascending": False}],
    max_results=1,
    output_format="list",
)
if not models:
    raise SystemExit("No models found in the experiment")

latest = models[0]
print(f"Latest model : {latest.model_id}")

model = mlflow.sklearn.load_model(f"models:/{latest.model_id}")
mlflow.sklearn.save_model(
    model, OUTPUT_DIR, skops_trusted_types=["sklearn.tree._tree.Tree"]
)
print(f"Model saved in ./{OUTPUT_DIR}")