import os
from PIL import Image, ImageDraw, ImageFont

IMAGE_DIR = os.path.join(os.path.dirname(__file__), "..", "frontend", "images")
os.makedirs(IMAGE_DIR, exist_ok=True)


def get_or_generate_city_image(city: str, aqi: float, aqi_level: str) -> str:
    """
    模拟 Amazon Bedrock 图像生成（仅用于本地演示）
    实际云部署时替换为 Bedrock API 调用
    """
    safe_name = "".join(c for c in city if c.isalnum() or c in (" ", "-", "_")).rstrip()
    filename = f"{safe_name.lower().replace(' ', '_')}_aqi_{int(aqi)}.png"
    filepath = os.path.join(IMAGE_DIR, filename)

    if os.path.exists(filepath):
        return filepath

    # 创建美观的占位图
    width, height = 600, 400
    img = Image.new("RGB", (width, height), color=(240, 248, 255))  # AliceBlue
    draw = ImageDraw.Draw(img)

    # 尝试加载更好的字体（可选）
    try:
        font_title = ImageFont.truetype("Arial Bold.ttf", 48)
        font_aqi = ImageFont.truetype("Arial.ttf", 32)
    except:
        font_title = ImageFont.load_default()
        font_aqi = ImageFont.load_default()

    # 绘制城市名（居中）
    bbox = draw.textbbox((0, 0), city, font=font_title)
    text_width = bbox[2] - bbox[0]
    x = (width - text_width) // 2
    draw.text((x, 80), city, fill=(30, 30, 30), font=font_title)

    # 绘制 AQI 信息
    aqi_text = f"AQI: {aqi:.0f} ({aqi_level})"
    bbox = draw.textbbox((0, 0), aqi_text, font=font_aqi)
    text_width = bbox[2] - bbox[0]
    x = (width - text_width) // 2
    draw.text((x, 160), aqi_text, fill=(50, 50, 50), font=font_aqi)

    # 添加装饰性元素（可选）
    draw.line([(50, 250), (width - 50, 250)], fill=(100, 100, 100), width=2)

    img.save(filepath)
    print(f"[Demo] Mock image saved: {filepath}")
    return filepath
