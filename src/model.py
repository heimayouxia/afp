import os
import pandas as pd
from autogluon.tabular import TabularPredictor

# 模拟从 "SageMaker Model Registry" 加载模型（实际为本地路径）
MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "data/ag_models")


class AQIPredictor:
    def __init__(self):
        print(f"Loading model from {MODEL_PATH}...")
        self.predictor = TabularPredictor.load(MODEL_PATH)
        # self.feature_columns = self.predictor.feature_metadata_inferred.features
        self.feature_columns = list(self.predictor.feature_metadata_in.get_features())

    def predict(self, city: str, date_str: str) -> dict:
        """
        模拟推理：输入城市和日期，返回 AQI 预测及等级
        """
        # 实际项目中，应调用特征存储（Feature Store）或实时计算特征
        # 为演示，构造一个符合训练 schema 的 dummy 样本
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
        # 真实场景需有特征工程 pipeline，此处简化为 mock 数据
        sample = pd.DataFrame(
            [
                {
                    "city": city,
                    "date": pd.to_datetime(date_str),
                    "TEMP": 36.5,
                    "DEWP": 33.1,
                    "SLP": 1008.7,
                    "STP": 969.2,
                    "VISIB": 8.3,
                    "WDSP": 12.9,
                    "MXSPD": 20,
                    "GUST": 26,
                    "MAX": 48.9,
                    "MIN": 26.1,
                    "PRCP": 0.19,
                    "SNDP": 0.0,
                    "PRCP": 0.0,
                    "Fog": 0,
                    "Rain": 1,
                    "Snow": 1,
                    "Hail": 0,
                    "Thunder": 0,
                    "Tornado": 0,
                }
            ]
        )

        # 预测 AQI 数值
        aqi_pred = self.predictor.predict(sample[feature_cols]).iloc[0]
        # aqi_pred = self.predictor.predict(sample).iloc[0]

        # EPA AQI 等级映射
        def aqi_to_level(aqi):
            if aqi <= 50:
                return "Good"
            elif aqi <= 100:
                return "Moderate"
            elif aqi <= 150:
                return "Unhealthy for Sensitive Groups"
            elif aqi <= 200:
                return "Unhealthy"
            elif aqi <= 300:
                return "Very Unhealthy"
            else:
                return "Hazardous"

        return {
            "city": city,
            "date": date_str,
            "predicted_aqi": round(float(aqi_pred), 1),
            "aqi_level": aqi_to_level(aqi_pred),
        }
