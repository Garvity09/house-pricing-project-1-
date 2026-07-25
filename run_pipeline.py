import os
import argparse
import logging
from generate_dataset import generate_olist_dataset
from pipelines.training_pipeline import train_pipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def main():
    parser = argparse.ArgumentParser(description="Run Customer Satisfaction MLOps Pipeline")
    parser.add_argument("--model", type=str, default="lightgbm", choices=["linear_regression", "random_forest", "xgboost", "lightgbm"], help="Model strategy to train")
    parser.add_argument("--data_path", type=str, default="data/olist_customers_dataset.zip", help="Path to zip or csv dataset")
    args = parser.parse_args()

    # Generate dataset if missing
    if not os.path.exists(args.data_path):
        print(f"Data path '{args.data_path}' not found. Generating synthetic dataset...")
        generate_olist_dataset()

    print(f"\n[+] Executing End-to-End MLOps Pipeline using model strategy: {args.model}\n")
    results = train_pipeline(data_path=args.data_path, model_name=args.model)

    print("\n" + "="*50)
    print("[RESULTS] PIPELINE EVALUATION SUMMARY RESULTS")
    print("="*50)

    print(f"Model Strategy : {results['model_name'].upper()}")
    print(f"MSE            : {results['metrics']['mse']:.4f}")
    print(f"RMSE           : {results['metrics']['rmse']:.4f}")
    print(f"R2 Score       : {results['metrics']['r2_score']:.4f}")
    print("="*50 + "\n")

if __name__ == "__main__":
    main()
