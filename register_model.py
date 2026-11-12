from azure.ai.ml import MLClient
from azure.ai.ml.entities import Model
from azure.ai.ml.constants import AssetTypes
from azure.identity import DefaultAzureCredential

credential = DefaultAzureCredential()
ml_client = MLClient.from_config(credential=credential)

# Find latest completed job
jobs = ml_client.jobs.list()
latest_job = None
for j in jobs:
    if j.status == "Completed" and j.display_name == "iris-rf-training":
        latest_job = j
        break

if latest_job is None:
    print("No completed jobs found.")
    exit(1)

print(f"Latest job: {latest_job.name}")

model = Model(
    name="iris-classifier-azure",
    description="Random Forest trained on Iris dataset",
    type=AssetTypes.CUSTOM_MODEL,
    path=f"azureml://jobs/{latest_job.name}/outputs/model_output/",
)

registered_model = ml_client.models.create_or_update(model)
print(f"Model registered: {registered_model.name}, version: {registered_model.version}")