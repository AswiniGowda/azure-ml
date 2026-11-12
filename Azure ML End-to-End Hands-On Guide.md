# Azure ML End-to-End Hands-On Guide

Nov 12, 2026 · @sarath

## Overview

This guide walks through a complete Azure ML workflow using the Python SDK v2: create a workspace, upload data, set up a custom environment, submit a training job, register the model, deploy it to a managed online endpoint, and test it. Every step is done programmatically — the only manual step is creating the workspace in the Azure Portal.

**Prerequisites:**

- Azure subscription with a resource group
- Azure CLI installed and logged in (`az login`)
- Python 3.11 with a virtual environment
- Packages: `pip install azure-ai-ml azure-identity`

## Project Structure

```
azure-ml/
├── config.json              ← downloaded from Azure Portal
├── data/
│   └── iris.csv             ← generated dataset
├── environment/
│   └── conda.yaml           ← custom environment definition
├── src/
│   ├── train.py             ← training script
│   └── score.py             ← scoring script for endpoint
├── generate_data.py         ← creates iris.csv (run once)
├── setup.py                 ← creates compute, data asset, environment
├── submit_job.py            ← submits training job
├── register_model.py        ← registers best model
├── deploy_endpoint.py       ← deploys and tests endpoint
└── test_endpoint.py         ← sends sample prediction request
```

**config.json** — downloaded from the workspace Overview page in Azure Portal. Contains subscription ID, resource group, and workspace name. `MLClient.from_config()` reads this automatically.

**environment/conda.yaml** — defines the Python packages installed on the remote compute. Key: pin `mlflow==2.15.1` and include `azureml-inference-server-http` for deployment.

**src/train.py** — standard sklearn training script. No Azure-specific code inside — just sklearn + mlflow for metrics + joblib for saving.

**src/score.py** — two functions: `init()` loads the model at startup, `run()` handles each prediction request. Azure ML wraps this into a web server automatically.

## Step 1: Setup — Workspace, Compute, Data, Environment

Create the workspace from Azure Portal (one-time). Everything else is SDK.

**generate\_data.py:**

```python
from sklearn.datasets import load_iris
import pandas as pd
import os

os.makedirs("data", exist_ok=True)
iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df["target"] = iris.target
df.to_csv("data/iris.csv", index=False)
print("iris.csv saved")
```

**environment/conda.yaml:**

```yaml
name: sklearn-env
channels:
  - conda-forge
  - defaults
dependencies:
  - python=3.11
  - pip
  - pip:
      - scikit-learn==1.5.2
      - mlflow==2.15.1
      - azureml-mlflow
      - azureml-inference-server-http
      - pandas
      - numpy
      - joblib
```

Pin `mlflow==2.15.1` — newer versions use the `LoggedModels` API which Azure ML's tracking server doesn't support, causing 404 errors on `log_model()`. Include `azureml-inference-server-http` — required for the scoring endpoint, not for training.

**setup.py:**

```python
from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential
from azure.ai.ml.entities import AmlCompute, Data, Environment
from azure.ai.ml.constants import AssetTypes

credential = DefaultAzureCredential()
ml_client = MLClient.from_config(credential=credential)
print(f"Connected to: {ml_client.workspace_name}")

# Compute cluster
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

# Data asset
iris_data = Data(
    name="iris-dataset",
    description="Iris dataset for training",
    path="./data/iris.csv",
    type=AssetTypes.URI_FILE,
)
ml_client.data.create_or_update(iris_data)
print("Dataset registered")

# Environment
env = Environment(
    name="sklearn-env-v3",
    description="Scikit-learn with MLflow 2.15.1",
    conda_file="./environment/conda.yaml",
    image="mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu22.04:latest",
)
ml_client.environments.create_or_update(env)
print("Environment created")
```

**What setup.py creates:**

- **Compute cluster** (`cpu-cluster`): Standard\_DS2\_v2 (2 cores, 7GB RAM), scales 0–2 instances. `min_instances=0` means no charges when idle.
- **Data asset** (`iris-dataset`): uploads `iris.csv` to Azure storage and registers it by name. Accessed in jobs as `azureml:iris-dataset:1`.
- **Environment** (`sklearn-env-v3`): base Docker image + conda.yaml. Azure ML builds a Docker image from these and caches it.

## Step 2: Training Script

**src/train.py:**

```python
import argparse
import os
import pandas as pd
import mlflow
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str)
    parser.add_argument("--n_estimators", type=int, default=100)
    parser.add_argument("--max_depth", type=int, default=5)
    parser.add_argument("--model_output", type=str, default="./outputs/model")
    args = parser.parse_args()

    df = pd.read_csv(args.data)
    X = df.drop("target", axis=1)
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestClassifier(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        random_state=42,
    )
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    f1 = f1_score(y_test, preds, average="weighted")

    mlflow.log_param("n_estimators", args.n_estimators)
    mlflow.log_param("max_depth", args.max_depth)
    mlflow.log_metric("accuracy", acc)
    mlflow.log_metric("f1_score", f1)

    os.makedirs(args.model_output, exist_ok=True)
    model_path = os.path.join(args.model_output, "model.pkl")
    joblib.dump(model, model_path)

    print(f"Accuracy: {acc:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"Model saved to: {model_path}")


if __name__ == "__main__":
    main()
```

**Key design decisions:**

- No `mlflow.autolog()` and no `mlflow.sklearn.log_model()` — these cause version conflicts with Azure ML's tracking server. Manual `log_param()` and `log_metric()` work reliably.
- Model saved with `joblib.dump()` to a named output directory (`--model_output`). Azure ML captures anything written there.
- No Azure-specific imports inside `train.py` — the same script runs locally or on any cloud platform.
- Arguments passed via command line so `submit_job.py` controls hyperparameters without editing the training script.

## Step 3: Submit Training Job

**submit\_job.py:**

```python
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
    environment="azureml:sklearn-env-v3@latest",
    compute="cpu-cluster",
    display_name="iris-rf-training",
    experiment_name="iris-azure-ml",
)

returned_job = ml_client.jobs.create_or_update(job)
print(f"Job submitted: {returned_job.name}")
print(f"Studio URL: {returned_job.studio_url}")
```

**What happens when you run this:**

1. Uploads `./src` folder to Azure storage
2. Spins up `cpu-cluster` (1–3 min if scaled to zero)
3. Builds Docker image from environment's conda.yaml (3–8 min, first time only; cached after)
4. Downloads `iris-dataset:1` into the container
5. Runs `python train.py --data <path> --n_estimators 150 --max_depth 7 --model_output <path>`
6. Training completes, metrics logged to Azure ML via mlflow
7. Model saved to the `model_output` folder, automatically uploaded to Azure storage
8. Compute scales back to zero

**Check in Studio:** Jobs → iris-azure-ml → click the latest run → Overview (params/metrics), Outputs + logs (logs and model files), Metrics tab.

## Step 4: Register Model Programmatically

**register\_model.py:**

```python
from azure.ai.ml import MLClient
from azure.ai.ml.entities import Model
from azure.ai.ml.constants import AssetTypes
from azure.identity import DefaultAzureCredential

credential = DefaultAzureCredential()
ml_client = MLClient.from_config(credential=credential)

# Find latest completed job dynamically
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
```

**Key points:**

- Uses `AssetTypes.CUSTOM_MODEL` (not `MLFLOW_MODEL`) because the model is saved with `joblib.dump()`, not `mlflow.sklearn.log_model()`.
- Path points to the named output `model_output` defined in `submit_job.py`'s `outputs` parameter.
- Dynamically finds the latest completed job — no hardcoded job names.
- Each run registers a new version (v1, v2, v3...) under the same model name.

## Step 5: Scoring Script and Deployment

**src/score.py:**

```python
import os
import json
import joblib
import numpy as np


def init():
    global model
    model_dir = os.environ["AZUREML_MODEL_DIR"]

    model_path = None
    for root, dirs, files in os.walk(model_dir):
        if "model.pkl" in files:
            model_path = os.path.join(root, "model.pkl")
            break

    if model_path is None:
        all_files = []
        for root, dirs, files in os.walk(model_dir):
            for f in files:
                all_files.append(os.path.join(root, f))
        raise FileNotFoundError(
            f"model.pkl not found in {model_dir}. Files: {all_files}"
        )

    print(f"Loading model from: {model_path}")
    model = joblib.load(model_path)
    print("Model loaded successfully")


def run(raw_data):
    data = json.loads(raw_data)
    features = np.array(data["data"])
    predictions = model.predict(features)
    return predictions.tolist()
```

`score.py` searches the entire model directory tree for `model.pkl` instead of assuming a fixed path. This handles Azure ML's nested directory structure reliably. If the file isn't found, it prints all files for debugging.

**deploy\_endpoint.py:**

```python
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

# Find latest model version dynamically
model_name = "iris-classifier-azure"
latest_model = None
for m in ml_client.models.list(name=model_name):
    if latest_model is None or int(m.version) > int(latest_model.version):
        latest_model = m

print(f"Using model: {latest_model.name}, version: {latest_model.version}")

# Create endpoint
endpoint = ManagedOnlineEndpoint(
    name="iris-endpoint",
    description="Iris prediction endpoint",
    auth_mode="key",
)
print("Creating endpoint...")
ml_client.online_endpoints.begin_create_or_update(endpoint).result()
print("Endpoint created")

# Create deployment
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

# Route 100% traffic
endpoint.traffic = {"iris-deploy": 100}
ml_client.online_endpoints.begin_create_or_update(endpoint).result()
print("Traffic routed 100% to iris-deploy")
```

**Deployment takes 5–10 minutes.** Azure ML builds a Docker image from the conda.yaml, starts a container with `score.py`, loads the registered model, and exposes an HTTP endpoint.

## Step 6: Test the Endpoint

**test\_endpoint.py:**

```python
from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential
import json

credential = DefaultAzureCredential()
ml_client = MLClient.from_config(credential=credential)

sample = {"data": [[5.1, 3.5, 1.4, 0.2]]}

with open("sample_request.json", "w") as f:
    json.dump(sample, f)

result = ml_client.online_endpoints.invoke(
    endpoint_name="iris-endpoint",
    request_file="sample_request.json",
)
print(f"Prediction: {result}")
```

Expected output: `[0]` (setosa). You can also test from Azure ML Studio: Endpoints → iris-endpoint → Test tab → paste `{"data": [[5.1, 3.5, 1.4, 0.2]]}` → click Test.

## Step 7: CI/CD with Azure DevOps

Automate the full pipeline: every push to `main` triggers train → register → deploy.

**azure-pipelines.yml:**

```yaml
trigger:
  branches:
    include:
      - main

pool:
  vmImage: "ubuntu-latest"

variables:
  azureServiceConnection: "azure-ml-connection"

steps:
  - task: UsePythonVersion@0
    inputs:
      versionSpec: "3.11"

  - script: pip install azure-ai-ml azure-identity
    displayName: "Install SDK"

  - task: AzureCLI@2
    displayName: "Train"
    inputs:
      azureSubscription: $(azureServiceConnection)
      scriptType: "bash"
      scriptLocation: "inlineScript"
      inlineScript: python submit_job.py --wait true

  - task: AzureCLI@2
    displayName: "Register Model"
    inputs:
      azureSubscription: $(azureServiceConnection)
      scriptType: "bash"
      scriptLocation: "inlineScript"
      inlineScript: python register_model.py

  - task: AzureCLI@2
    displayName: "Deploy"
    inputs:
      azureSubscription: $(azureServiceConnection)
      scriptType: "bash"
      scriptLocation: "inlineScript"
      inlineScript: python deploy_endpoint.py
```

**Setup required in Azure DevOps:**

1. Create a service connection: Project Settings → Service connections → Azure Resource Manager → Service principal (automatic)
2. Name it `azure-ml-connection`, select your subscription and resource group
3. Push `azure-pipelines.yml` to your repo

**How it works:** Azure DevOps VMs come with Azure CLI pre-installed. The `AzureCLI@2` task authenticates using the service connection (service principal), then runs your Python scripts. `DefaultAzureCredential()` in your code picks up the token automatically. Same scripts locally and in CI/CD — no code changes.

**For `--wait true` support**, add argument parsing to `submit_job.py`:

```python
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--wait", type=str, default="false")
args = parser.parse_args()

# ... after job submission:
if args.wait.lower() == "true":
    print("Waiting for job to complete...")
    ml_client.jobs.stream(returned_job.name)
    completed_job = ml_client.jobs.get(returned_job.name)
    if completed_job.status != "Completed":
        exit(1)
```

## Cleanup and Cost Management

**Delete endpoint (charges while running):**

```python
ml_client.online_endpoints.begin_delete(name="iris-endpoint").result()
```

**Delete compute cluster:**

```python
ml_client.compute.begin_delete(name="cpu-cluster").result()
```

**Delete entire resource group (removes everything):**

```bash
az group delete --name hands-on-azure --yes
```

**What costs money:**

- Managed online endpoints — billed per hour while deployed, even with zero traffic
- Compute cluster — billed when instances are running. `min_instances=0` stops charges when idle
- Storage — minimal cost for datasets and model files

**Always delete endpoints immediately after testing.** Compute clusters with `min_instances=0` scale down automatically, but endpoints don't.

## Key Interview Points

**"How do you train models on Azure ML?"** — I use the Python SDK v2 to submit command jobs. The training script is cloud-agnostic (standard sklearn + mlflow), and all Azure wiring — compute, data, environment — is handled by the job submission script. The same train.py runs locally and on Azure.

**"How do environments work?"** — Azure ML takes a base Docker image and a conda.yaml, builds a Docker image with those packages, and caches it. First run builds the image (5–10 min), subsequent runs reuse the cache. Every job runs inside a Docker container.

**"How do you register models?"** — Programmatically using the SDK. The training job saves the model to a named output directory, and the registration script finds the latest completed job and registers the model from that output path. Each registration creates a new version.

**"How do you deploy models?"** — Managed online endpoints with a scoring script (score.py) that defines init() and run(). Azure ML wraps this into a web server, loads the registered model, and exposes an HTTP endpoint. I use blue/green deployments in production — deploy v2 alongside v1, shift traffic gradually.

**"How do you automate this?"** — Azure DevOps pipeline with three steps: train (submit job + wait), register (find latest completed job), deploy (create/update endpoint). Same Python scripts used locally, authenticated through a service connection.

**"Why not use mlflow.autolog() for model saving?"** — autolog works for params and metrics, but mlflow.sklearn.log\_model() has version conflicts with Azure ML's tracking server. The reliable pattern is autolog(log\_models=False) for tracking + joblib.dump() for saving the model to a named output directory.

**Run order for the complete workflow:**

```bash
python generate_data.py    # one time
python setup.py            # one time
python submit_job.py       # per training run
python register_model.py   # per model version
python deploy_endpoint.py  # per deployment
python test_endpoint.py    # verify
```
