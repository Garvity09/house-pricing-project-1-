import logging
import os
import joblib
from pipelines.training_pipeline import train_pipeline

def continuous_deployment_pipeline(data_path: str, min_accuracy: float = 0.50, model_name: str = "lightgbm"):
    """Deployment pipeline that trains a model and deploys it if minimum threshold condition is satisfied."""
    logging.info("Starting Continuous Deployment Pipeline...")
    
    result = train_pipeline(data_path=data_path, model_name=model_name)
    r2_score = result["metrics"]["r2_score"]
    
    if r2_score >= min_accuracy:
        logging.info(f"Model accuracy ({r2_score:.4f}) meets minimum deployment threshold ({min_accuracy}). Deploying model...")
        os.makedirs("deployed_model", exist_ok=True)
        deployed_path = os.path.join("deployed_model", "satisfaction_model.pkl")
        joblib.dump(result["model"], deployed_path)
        
        # Save feature list for Streamlit inference app
        feature_names = list(result["X_train"].columns)
        joblib.dump(feature_names, os.path.join("deployed_model", "feature_names.pkl"))
        
        logging.info(f"Model successfully deployed to {deployed_path}")
        return True, deployed_path, result["metrics"]
    else:
        logging.warning(f"Model accuracy ({r2_score:.4f}) did NOT meet minimum deployment threshold ({min_accuracy}). Deployment aborted.")
        return False, None, result["metrics"]
