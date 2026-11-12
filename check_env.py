from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential

credential = DefaultAzureCredential()
ml_client = MLClient.from_config(credential=credential)

for env in ml_client.environments.list():
    print(f"{env.name} : {env.version}")