# 🏡 House Pricing Prediction & Deployment Pipeline

[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/)
[![Framework: ZenML](https://img.shields.io/badge/MLOps-ZenML-7B2CBF.svg)](https://zenml.io/)
[![Tracking: MLflow](https://img.shields.io/badge/Tracking-MLflow-0194E2.svg)](https://mlflow.org/)
[![UI: Streamlit](https://img.shields.io/badge/UI-Streamlit-FF4B4B.svg)](https://streamlit.io/)
[![Deployment: Vercel / FastAPI](https://img.shields.io/badge/Deployment-FastAPI%20%7C%20Vercel-000000.svg)](https://vercel.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> An end-to-end MLOps pipeline for housing price prediction built with **ZenML**, **MLflow**, **Streamlit**, and a dynamic web frontend. Standardized for automated data ingestion, model training, evaluation, and continuous deployment.

---

## 🎯 Key Features

- 🔄 **Production MLOps Pipeline**: Orchestrated data ingestion, cleaning, model training, evaluation, and deployment using **ZenML**.
- 📊 **Experiment Tracking & Logging**: Integrated **MLflow** for tracking hyperparameter runs, metric logging, and artifact management.
- 🤖 **Multiple ML Architectures**: Evaluated Linear Regression, Random Forest, LightGBM, and XGBoost regressor models.
- 🌐 **Interactive Web UI**: Includes both a Python Streamlit app and a lightweight Vercel-ready HTML5/CSS3/JavaScript frontend for real-time inference.
- ⚙️ **Clean Modular Architecture**: Clean separation of pipeline steps (`steps/`), source abstractions (`src/`), and deployment triggers (`run_deployment.py`).

---

## 🏗️ Architecture & Pipeline Overview

```
                   +------------------------+
                   |  Raw Dataset Ingestion |
                   +-----------+------------+
                               |
                               v
                   +------------------------+
                   | Data Cleaning & Prep   |
                   +-----------+------------+
                               |
                               v
                   +------------------------+
                   |  Model Training (SKL)  |
                   +-----------+------------+
                               |
                               v
                   +------------------------+
                   | Model Evaluation (MSE) |
                   +-----------+------------+
                               |
                               v
                   +------------------------+
                   | MLflow Serving Deploy  |
                   +-----------+------------+
                               |
                               v
                 +----------------------------+
                 | Streamlit & Web Frontend   |
                 +----------------------------+
```

---

## 📁 Project Structure

```text
house-pricing-project-1-/
├── app.py                      # Streamlit prediction application
├── generate_dataset.py         # Synthetic housing data generator
├── run_pipeline.py             # Script to trigger training pipeline
├── run_deployment.py           # Script to launch deployment server
├── requirements.txt            # Project dependencies
├── vercel.json                 # Vercel web deployment configuration
├── pipelines/
│   ├── deployment_pipeline.py  # Continuous deployment & inference pipeline
│   └── training_pipeline.py    # Training pipeline orchestration
├── src/
│   ├── data_cleaning.py        # Data preprocessing & feature engineering
│   ├── data_ingestion.py       # Data loader helper utilities
│   ├── evaluation.py          # Regression evaluation metrics (MSE, RMSE, R²)
│   └── model_dev.py            # Model training abstraction layers
├── steps/
│   ├── clean_data.py           # ZenML step for cleaning
│   ├── config.py               # Step parameters & configuration
│   ├── evaluation.py          # ZenML step for model evaluation
│   ├── ingest_data.py          # ZenML step for data loading
│   └── model_train.py          # ZenML step for model training
└── web/
    ├── app.js                  # Frontend interactive scripts
    ├── index.html              # Modern web UI template
    └── style.css               # Modern styling tokens
```

---

## 🚀 Quickstart & Installation

### 1. Prerequisites
Ensure Python **3.10+** is installed on your system.

### 2. Install Dependencies
```bash
git clone https://github.com/Garvity09/house-pricing-project-1-.git
cd house-pricing-project-1-
pip install -r requirements.txt
```

### 3. Initialize ZenML Stack
```bash
zenml init
zenml integration install mlflow -y
```

### 4. Run Training Pipeline
```bash
python run_pipeline.py
```

### 5. Launch Continuous Deployment & Streamlit Web App
```bash
# Run deployment pipeline
python run_deployment.py

# Launch interactive Streamlit dashboard
streamlit run app.py
```

---

## 📊 Evaluation Metrics

Models are benchmarked using key regression metrics:
- **Mean Squared Error (MSE)**
- **Root Mean Squared Error (RMSE)**
- **R-Squared ($R^2$) Score**

Log artifacts and hyperparameter runs are accessible locally via MLflow UI:
```bash
mlflow ui --backend-store-uri <path-to-mlruns>
```

---

## 📜 License

Distributed under the MIT License. See `LICENSE` for details.
