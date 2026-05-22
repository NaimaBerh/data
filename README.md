# FakeGuard AI

Unified Fake Account Detection Model for GitHub, LinkedIn, and Instagram using XGBoost.

## Overview

FakeGuard AI detects fake accounts across three major platforms:
- **GitHub** (platform_id: 2)
- **LinkedIn** (platform_id: 0)  
- **Instagram** (platform_id: 1)

The model uses a unified XGBoost classifier trained on combined data from all platforms with SMOTE for handling class imbalance.

## Features

- **Data Integration**: Aligns features from different platforms into a common feature set
- **Platform Identification**: Uses `platform_id` to learn platform-specific patterns
- **Imbalanced Data Handling**: SMOTE oversampling for balanced training
- **Comprehensive Evaluation**: Accuracy, Precision, Recall, F1-Score, ROC-AUC
- **Platform-wise Metrics**: Performance breakdown by each platform
- **Easy Inference**: Simple `predict_fake()` function
- **API Ready**: FastAPI deployment included

## Installation

```bash
pip install xgboost imbalanced-learn scikit-learn pandas numpy
pip install fastapi uvicorn  # For API deployment
```

## Usage

### Training the Model

```bash
python fakeguard_ai.py
```

This will:
1. Load and preprocess datasets from all three platforms
2. Align features to a common set
3. Add platform IDs
4. Apply SMOTE for class balancing
5. Train XGBoost model
6. Evaluate performance
7. Save model to `fakeguard_model.pkl`

### Using the Model for Predictions

```python
from fakeguard_ai import load_model, predict_fake

# Load the trained model
model_data = load_model('fakeguard_model.pkl')

# Example: Predict a LinkedIn account
linkedin_account = {
    'Followers': 500,
    'Connections': 250,
    'Photo': 1,
    'Number of Skills': 10,
    'Number of Experiences': 3
}

result = predict_fake(model_data, linkedin_account, platform='LinkedIn')
print(f"Label: {result['label']}")
print(f"Confidence: {result['confidence']:.2%}")
print(f"Probabilities: {result['probabilities']}")
```

### Deploying as API

```bash
# Start the FastAPI server
python api_server.py

# Or with uvicorn directly
uvicorn api_server:app --host 0.0.0.0 --port 8000
```

#### API Endpoints

- `GET /` - Welcome message and version info
- `GET /health` - Health check
- `POST /predict` - Single account prediction
- `POST /predict/batch` - Batch predictions
- `GET /features` - Get required feature list

#### Example API Request

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "features": {
      "Followers": 500,
      "Connections": 250,
      "Photo": 1,
      "Number of Skills": 10
    },
    "platform": "LinkedIn"
  }'
```

## Model Performance

The unified model achieves:
- **Overall Accuracy**: ~98.5%
- **Precision**: ~99.3%
- **Recall**: ~99.1%
- **F1-Score**: ~99.2%
- **ROC-AUC**: ~98.8%

### Platform-wise Performance
- **LinkedIn**: ~100% Accuracy
- **Instagram**: ~96% Accuracy
- **GitHub**: ~98% Accuracy

## Dataset Format

### LinkedIn (`LinkedIn_Dataset.csv`)
Contains profile features like connections, followers, experiences, skills, etc.
Label column: `test` (mapped: 0,1 → Real; 10,11 → Fake)

### Instagram (`instagram_dataset.csv`)
Contains user metrics like follower count, following count, media count, etc.
Label column: `isFake` (inverted: 0 → Real, 1 → Fake)

### GitHub (`bothawk_data_Ori.csv`)
Contains activity metrics like commits, repositories, issues, etc.
Label column: `label` (Human → Real, Bot → Fake)

## Files

- `fakeguard_ai.py` - Main training and prediction pipeline
- `api_server.py` - FastAPI deployment server
- `fakeguard_model.pkl` - Saved trained model (generated after training)
- `README.md` - This file

## License

MIT License
