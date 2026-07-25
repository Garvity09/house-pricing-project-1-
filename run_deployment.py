import os
import argparse
import logging
from generate_dataset import generate_olist_dataset
from pipelines.deployment_pipeline import continuous_deployment_pipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def main():
    parser = argparse.ArgumentParser(description="Run Continuous Deployment Pipeline")
    parser.add_argument("--min_accuracy", type=float, default=0.20, help="Minimum R2 score required for deployment")
    parser.add_argument("--model", type=str, default="lightgbm", help="Model to train and deploy")
    parser.add_argument("--data_path", type=str, default="data/olist_customers_dataset.zip", help="Dataset path")
    args = parser.parse_args()

    if not os.path.exists(args.data_path):
        generate_olist_dataset()

    print(f"\n[+] Running Deployment Pipeline (Min threshold R2: {args.min_accuracy})...\n")
    deployed, model_path, metrics = continuous_deployment_pipeline(
        data_path=args.data_path,
        min_accuracy=args.min_accuracy,
        model_name=args.model
    )

    if deployed:
        print(f"\n[SUCCESS] Model successfully trained & deployed to {model_path}!")
        print(f"Metrics -> MSE: {metrics['mse']:.4f}, RMSE: {metrics['rmse']:.4f}, R2: {metrics['r2_score']:.4f}")
    else:
        print(f"\n[WARNING] Model metrics were below minimum criteria.")


if __name__ == "__main__":
    main()
