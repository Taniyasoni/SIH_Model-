"""
Upload Groundwater Forecasting Model to Hugging Face
"""
from huggingface_hub import HfApi, create_repo
import shutil
import os

# Configuration
REPO_NAME = "groundwater-forecaster"  # Change this to your desired repo name
USERNAME = "Taniyasoni"  # Your Hugging Face username

# Create temporary folder for upload
temp_folder = "temp_hf_upload"
if os.path.exists(temp_folder):
    shutil.rmtree(temp_folder)
os.makedirs(temp_folder)

print("📦 Copying model files...")

# Copy necessary directories
directories_to_copy = ["model", "scalers", "mappings", "metadata", "inference"]
for dir_name in directories_to_copy:
    if os.path.exists(dir_name):
        shutil.copytree(dir_name, os.path.join(temp_folder, dir_name))
        print(f"✓ Copied {dir_name}/")
    else:
        print(f"⚠ Warning: {dir_name}/ not found")

# Copy important files
files_to_copy = ["requirements.txt", "main.py", "README.md"]
for file_name in files_to_copy:
    if os.path.exists(file_name):
        shutil.copy(file_name, temp_folder)
        print(f"✓ Copied {file_name}")
    else:
        print(f"⚠ Warning: {file_name} not found")

# Create model card (README) if it doesn't exist
readme_path = os.path.join(temp_folder, "README.md")
if not os.path.exists(readme_path):
    with open(readme_path, "w") as f:
        f.write("""---
license: mit
tags:
- groundwater
- forecasting
- ensemble
- pytorch
- time-series
---

# Groundwater Level Forecasting Model

3-model ensemble approach for 30-day groundwater level forecasting.

## Model Description

This model uses an ensemble of three components to predict groundwater levels:
- Historical pattern recognition
- Weather influence modeling
- Trend analysis

## Usage

```python
from inference.gw_ensemble_forecaster import GroundwaterEnsembleForecaster
import numpy as np
from pathlib import Path

# Load the model
forecaster = GroundwaterEnsembleForecaster(Path("."))

# Make prediction
result = forecaster.predict(
    station_id=1001,
    gw_history_7days=np.array([10.5, 10.3, 10.2, 10.1, 10.0, 9.9, 9.8]),
    rainfall_30d=50.0,
    temp_mean=25.0,
    temp_range=10.0,
    pet_7d=35.0
)

print(f"30-day forecast: {result['forecast_30day']}")
print(f"Mean: {result['forecast_mean']}")
print(f"Confidence: {result['confidence']}")
```

## API Deployment

Can be deployed as a FastAPI service using `main.py`.

## Requirements

See `requirements.txt` for dependencies.
""")
    print("✓ Created README.md")

print("\n🚀 Uploading to Hugging Face...")

try:
    # Initialize API
    api = HfApi()
    
    # Create repository (will skip if already exists)
    repo_id = f"{USERNAME}/{REPO_NAME}"
    try:
        create_repo(repo_id=repo_id, repo_type="model", exist_ok=True)
        print(f"✓ Repository created/verified: {repo_id}")
    except Exception as e:
        print(f"ℹ Repository may already exist: {e}")
    
    # Upload folder
    print("📤 Uploading files...")
    api.upload_folder(
        folder_path=temp_folder,
        repo_id=repo_id,
        repo_type="model",
        commit_message="Upload groundwater forecasting model"
    )
    
    print(f"\n✅ SUCCESS! Model uploaded to: https://huggingface.co/{repo_id}")
    
except Exception as e:
    print(f"\n❌ Error uploading: {e}")
    print("\nMake sure you're logged in: huggingface-cli login")

finally:
    # Cleanup
    print("\n🧹 Cleaning up temporary files...")
    if os.path.exists(temp_folder):
        shutil.rmtree(temp_folder)
    print("✓ Done!")
