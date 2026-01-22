import os
from diffusers import StableDiffusionPipeline
import torch
from PIL import Image

# 使用开源模型
MODEL_NAME = "stabilityai/stable-diffusion-2-1-base"
IMAGE_DIR = os.path.join(os.path.dirname(__file__), "..", "frontend", "images")

# 创建输出目录
os.makedirs(IMAGE_DIR, exist_ok=True)


class CityImageGenerator:
    def __init__(self, use_cpu=True):
        print(f"Loading {MODEL_NAME} (this may take a while on CPU)...")
        self.pipe = StableDiffusionPipeline.from_pretrained(
            MODEL_NAME,
            torch_dtype=torch.float16 if not use_cpu else torch.float32,
            safety_checker=None,  # 关闭安全检查以简化演示（生产环境应开启）
        )
        if not use_cpu and torch.cuda.is_available():
            self.pipe = self.pipe.to("cuda")
        else:
            self.pipe = self.pipe.to("cpu")

    def generate_image(self, city: str, aqi: float, aqi_level: str) -> str:
        """
        生成城市空气质量主题图，返回保存路径
        """
        # 构造提示词（Prompt Engineering）
        prompt = (
            f"A photorealistic view of {city} on a clear day, "
            f"with clean air and blue sky, "
            f"AQI: {aqi:.0f} ({aqi_level}), "
            f"environmental health, high detail, 4k"
        )

        # 生成图像
        image = self.pipe(prompt, num_inference_steps=20).images[0]

        # 保存
        filename = f"{city.replace(' ', '_').lower()}_aqi_{int(aqi)}.png"
        filepath = os.path.join(IMAGE_DIR, filename)
        image.save(filepath)
        print(f"Generated image saved to {filepath}")
        return filepath


# 提供预生成图片回退机制
def get_or_generate_city_image(city: str, aqi: float, aqi_level: str) -> str:
    """优先返回已存在图片，避免重复生成（节省时间）"""
    filename = f"{city.replace(' ', '_').lower()}_aqi_{int(aqi)}.png"
    filepath = os.path.join(IMAGE_DIR, filename)
    if os.path.exists(filepath):
        return filepath
    else:
        generator = CityImageGenerator(use_cpu=True)
        return generator.generate_image(city, aqi, aqi_level)
