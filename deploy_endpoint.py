from azure.ai.ml import MLClient
from azure.ai.ml.entities import (
    ManagedOnlineEndpoint,
    ManagedOnlineDeployment,
    CodeConfiguration,
    Environment,
)
from azure.identity import DefaultAzureCredential

credential = DefaultAzureCredential()
ml_client = MLClient.from_config(credential=credential)

# --- Find latest model version ---
model_name = "iris-classifier-azure"
latest_model = None
for m in ml_client.models.list(name=model_name):
    if latest_model is None or int(m.version) > int(latest_model.version):
        latest_model = m

print(f"Using model: {latest_model.name}, version: {latest_model.version}")

# --- Create endpoint ---
endpoint = ManagedOnlineEndpoint(
    name="iris-endpoint",
    description="Iris prediction endpoint",
    auth_mode="key",
)
print("Creating endpoint...")
ml_client.online_endpoints.begin_create_or_update(endpoint).result()
print("Endpoint created")

# --- Create deployment ---
deployment = ManagedOnlineDeployment(
    name="iris-deploy",
    endpoint_name="iris-endpoint",
    model=f"azureml:{model_name}:{latest_model.version}",
    code_configuration=CodeConfiguration(
        code="./src",
        scoring_script="score.py",
    ),
    environment=Environment(
        conda_file="./environment/conda.yaml",
        image="mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu22.04:latest",
    ),
    instance_type="Standard_DS2_v2",
    instance_count=1,
)
print("Creating deployment...")
ml_client.online_deployments.begin_create_or_update(deployment).result()
print("Deployment created")

# --- Route traffic ---
endpoint.traffic = {"iris-deploy": 100}
ml_client.online_endpoints.begin_create_or_update(endpoint).result()
print("Traffic routed 100% to iris-deploy")