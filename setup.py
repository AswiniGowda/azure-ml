from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential
from azure.ai.ml.entities import AmlCompute, Data, Environment
from azure.ai.ml.constants import AssetTypes

credential = DefaultAzureCredential()
ml_client = MLClient.from_config(credential=credential)
print(f"Connected to: {ml_client.workspace_name}")

# --- Compute ---
try:
    compute = ml_client.compute.get("cpu-cluster")
    print("Compute cluster already exists")
except Exception:
    compute = AmlCompute(
        name="cpu-cluster",
        type="amlcompute",
        size="Standard_DS2_v2",
        min_instances=0,
        max_instances=2,
    )
    ml_client.compute.begin_create_or_update(compute).result()
    print("Compute cluster created")

# --- Data ---
iris_data = Data(
    name="iris-dataset",
    description="Iris dataset for training",
    path="./data/iris.csv",
    type=AssetTypes.URI_FILE,
)
ml_client.data.create_or_update(iris_data)
print("Dataset registered")

# --- Environment ---
env = Environment(
    name="sklearn-mlflow-env-v2",
    description="Scikit-learn with MLflow 2.16.2",
    conda_file="./environment/conda.yaml",
    image="mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu22.04:latest",
)
ml_client.environments.create_or_update(env)
print("Environment created")

print("\nSetup complete!")