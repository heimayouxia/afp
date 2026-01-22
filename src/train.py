import os
import sys
import time
import json
import logging
from datetime import datetime

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from sklearn.model_selection import train_test_split
from autogluon.tabular import TabularPredictor

# 设置日志
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


# 确保目录存在
def setup_dirs():
    dirs = [
        os.path.join(os.path.dirname(__file__), "../data/processed/"),
        os.path.join(os.path.dirname(__file__), "../data/ag_models/"),
        os.path.join(os.path.dirname(__file__), "../results/"),
    ]
    for d in dirs:
        os.makedirs(d, exist_ok=True)


def read_and_split():
    csv_path = os.path.join(
        os.path.dirname(__file__), "../", "data/processed/noaa_openaq_aqi_frshtt.csv"
    )
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Input data not found: {csv_path}")

    df = pd.read_csv(csv_path)
    train_df, val_df = train_test_split(df, test_size=0.15, random_state=42)

    train_csv_path = os.path.join(
        os.path.dirname(__file__), "../", "data/processed/train_dataset.csv"
    )
    val_csv_path = os.path.join(
        os.path.dirname(__file__), "../", "data/processed/val_dataset.csv"
    )

    train_df.to_csv(train_csv_path, index=False)
    val_df.to_csv(val_csv_path, index=False)
    logging.info(
        f"Data split: {len(train_df)} train, {len(val_df)} validation samples."
    )


def main():
    setup_dirs()
    read_and_split()

    # 1. 加载数据
    train_dataset_path = os.path.join(
        os.path.dirname(__file__), "../", "data/processed/train_dataset.csv"
    )
    val_dataset_path = os.path.join(
        os.path.dirname(__file__), "../", "data/processed/val_dataset.csv"
    )
    train_df = pd.read_csv(train_dataset_path)
    val_df = pd.read_csv(val_dataset_path)

    feature_cols = [
        "TEMP",
        "DEWP",
        "SLP",
        "STP",
        "VISIB",
        "WDSP",
        "MXSPD",
        "GUST",
        "MAX",
        "MIN",
        "PRCP",
        "SNDP",
        "Fog",
        "Rain",
        "Snow",
        "Hail",
        "Thunder",
        "Tornado",
    ]
    label_col = "max_aqi"

    # 验证列是否存在
    missing_cols = [
        col for col in feature_cols + [label_col] if col not in train_df.columns
    ]
    if missing_cols:
        raise ValueError(f"Missing columns in data: {missing_cols}")

    # 2. 模型配置
    hyperparams = {
        "GBM": {},  # LightGBM
        "XGB": {"learning_rate": [0.01, 0.1], "max_depth": [3, 6, 9]},
        "CAT": {},
        "RF": {"n_estimators": 100},
        "NN_TORCH": {},
    }

    model_save_path = os.path.join(os.path.dirname(__file__), "../", "data/ag_models/")
    predictor = TabularPredictor(
        label=label_col,
        path=model_save_path,
        problem_type="regression",
        eval_metric="rmse",
    )

    # 3. 训练
    logging.info("Starting training...")
    start_time = time.time()
    predictor.fit(
        train_data=train_df[feature_cols + [label_col]],
        hyperparameters=hyperparams,
        time_limit=60,  # 秒
        presets="best_quality",
        num_cpus=4,
    )
    train_time = time.time() - start_time
    logging.info(f"Training completed in {train_time:.2f} seconds.")

    # 4. Leaderboard
    print("\n*** Leaderboard ***")
    lb = predictor.leaderboard()
    print(lb)

    # 5. 评估
    y_true = val_df[label_col]
    y_pred = predictor.predict(val_df)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    print("\n*** Performance Metrics ***")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE:  {mae:.4f}")
    print(f"R²:   {r2:.4f}")

    # 6. 预测 vs 真实
    plt.figure(figsize=(8, 6))
    plt.scatter(y_true, y_pred, alpha=0.6, edgecolors="k", s=20)
    plt.xlim(0, 800)
    plt.ylim(0, 800)
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], "r--", lw=2)
    plt.xlabel("True AQI")
    plt.ylabel("Predicted AQI")
    plt.title("Prediction vs True (Validation Set)")
    pred_vs_true_path = os.path.join(
        os.path.dirname(__file__), "../results/pred_vs_true.png"
    )
    plt.savefig(pred_vs_true_path, dpi=150, bbox_inches="tight")
    plt.close()

    # 7. Residual 图
    residuals = y_true - y_pred
    plt.figure(figsize=(8, 4))
    plt.hist(residuals, bins=90, color="skyblue", edgecolor="black")
    # plt.hist(residuals, bins=30, color='skyblue', edgecolor='black')
    plt.xlabel("Residuals (True - Pred)")
    plt.ylabel("Frequency")
    plt.title("Residual Distribution")
    plt.xlim(-300, 300)
    residuals_path = os.path.join(os.path.dirname(__file__), "../results/residuals.png")
    plt.savefig(residuals_path, dpi=150, bbox_inches="tight")
    plt.close()

    # 8. 特征重要性
    try:
        feat_imp = predictor.feature_importance(val_df)
        feat_imp_path = os.path.join(
            os.path.dirname(__file__), "../results/feature_importance.csv"
        )
        feat_imp.to_csv(feat_imp_path)
        print("\n*** Top 10 Feature Importances ***")
        print(feat_imp.head(10))
    except Exception as e:
        logging.warning(f"Feature importance failed: {e}")

    # 9. 实验日志
    results_log = {
        "timestamp": datetime.now().isoformat(),
        "train_samples": len(train_df),
        "val_samples": len(val_df),
        "features": feature_cols,
        "label": label_col,
        "time_limit_sec": 60,
        "actual_train_time_sec": round(train_time, 2),
        "best_model": predictor.model_best,
        "performance": {"rmse": float(rmse), "mae": float(mae), "r2": float(r2)},
        "leaderboard_shape": lb.shape,
    }

    log_path = os.path.join(os.path.dirname(__file__), "../results/experiment_log.json")
    with open(log_path, "w") as f:
        json.dump(results_log, f, indent=4)
    logging.info(f"Experiment log saved to {log_path}")

    # 10. 保存模型
    predictor.save()
    logging.info("Model saved successfully.")
    print("Training and evaluation completed!")


if __name__ == "__main__":
    main()
