# 示例：端到端调用
from src.model import AQIPredictor
from src.genai import get_or_generate_city_image
# from src.genai_sd import get_or_generate_city_image

# 1. 预测
pred = AQIPredictor().predict("Los Angeles", "2026-01-21")
print(
    pred
)  # {'city': 'Los Angeles', 'date': '2026-01-21', 'predicted_aqi': 30.1, 'aqi_level': 'Good'}

# 2. 生成图
img_path = get_or_generate_city_image(
    pred["city"], pred["predicted_aqi"], pred["aqi_level"]
)
print(f"Image at: {img_path}")
