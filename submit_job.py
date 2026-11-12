from azure.ai.ml import MLClient, command, Input, Output
from azure.identity import DefaultAzureCredential

credential = DefaultAzureCredential()
ml_client = MLClient.from_config(credential=credential)

job = command(
    code="./src",
    command="python train.py --data ${{inputs.data}} --n_estimators 150 --max_depth 7 --model_output ${{outputs.model_output}}",
    inputs={
        "data": Input(
            type="uri_file",
            path="azureml:iris-dataset:1",
        )
    },
    outputs={
        "model_output": Output(
            type="uri_folder",
            mode="rw_mount",
        )
    },
    environment="azureml:sklearn-mlflow-env-v2:1",
    compute="cpu-cluster",
    display_name="iris-rf-training",
    experiment_name="iris-azure-ml",
)

returned_job = ml_client.jobs.create_or_update(job)
print(f"Job submitted: {returned_job.name}")
print(f"Studio URL: {returned_job.studio_url}")