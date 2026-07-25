import os
import zipfile
import numpy as np
import pandas as pd

def generate_olist_dataset(data_dir: str = "data", n_samples: int = 5000):
    os.makedirs(data_dir, exist_ok=True)
    csv_path = os.path.join(data_dir, "olist_customers_dataset.csv")
    zip_path = os.path.join(data_dir, "olist_customers_dataset.zip")

    np.random.seed(42)

    payment_sequential = np.random.randint(1, 4, n_samples)
    payment_installments = np.random.randint(1, 10, n_samples)
    payment_value = np.round(np.random.exponential(scale=120, size=n_samples) + 15, 2)
    price = np.round(payment_value * np.random.uniform(0.7, 0.9, n_samples), 2)
    freight_value = np.round(payment_value - price, 2)
    freight_value = np.maximum(freight_value, 5.0)

    product_name_lenght = np.random.randint(10, 70, n_samples)
    product_description_lenght = np.random.randint(50, 1500, n_samples)
    product_photos_qty = np.random.randint(1, 8, n_samples)
    product_weight_g = np.random.randint(100, 10000, n_samples)
    product_length_cm = np.random.randint(10, 80, n_samples)
    product_height_cm = np.random.randint(5, 50, n_samples)
    product_width_cm = np.random.randint(10, 60, n_samples)

    # Customer and seller distance metrics & days
    delivery_days = np.random.randint(2, 25, n_samples)
    estimated_delivery_days = delivery_days + np.random.randint(-3, 10, n_samples)
    delay_days = np.maximum(0, delivery_days - estimated_delivery_days)
    
    # Review score generation based on delivery speed and price
    score_noise = np.random.normal(0, 0.6, n_samples)
    raw_score = 5 - (delay_days * 0.4) - (delivery_days * 0.05) + score_noise
    review_score = np.clip(np.round(raw_score), 1, 5).astype(int)

    df = pd.DataFrame({
        "order_id": [f"ord_{i:06d}" for i in range(n_samples)],
        "customer_id": [f"cust_{i:06d}" for i in range(n_samples)],
        "payment_sequential": payment_sequential,
        "payment_installments": payment_installments,
        "payment_value": payment_value,
        "price": price,
        "freight_value": freight_value,
        "product_name_lenght": product_name_lenght,
        "product_description_lenght": product_description_lenght,
        "product_photos_qty": product_photos_qty,
        "product_weight_g": product_weight_g,
        "product_length_cm": product_length_cm,
        "product_height_cm": product_height_cm,
        "product_width_cm": product_width_cm,
        "delivery_days": delivery_days,
        "estimated_delivery_days": estimated_delivery_days,
        "delay_days": delay_days,
        "review_score": review_score
    })

    # Save CSV
    df.to_csv(csv_path, index=False)
    print(f"Generated dataset with {n_samples} rows saved to: {csv_path}")

    # Compress into Zip
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
        zipf.write(csv_path, os.path.basename(csv_path))
    print(f"Compressed dataset into zip archive: {zip_path}")

    return zip_path

if __name__ == "__main__":
    generate_olist_dataset()
