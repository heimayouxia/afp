from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware  # ← 新增导入
from pydantic import BaseModel
from .model import AQIPredictor

app = FastAPI(
    title="Air Quality Prediction API",
    description="Simulates an AWS SageMaker Endpoint for AQI forecasting",
)

# 添加 CORS 中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 允许所有来源（演示用，生产环境应限制）
    allow_credentials=True,
    allow_methods=["*"],  # 允许所有方法（GET, POST, OPTIONS 等）
    allow_headers=["*"],  # 允许所有头
)

# 全局加载模型
predictor = AQIPredictor()


class PredictionRequest(BaseModel):
    city: str
    date: str


@app.post("/predict")
async def predict(request: PredictionRequest):
    try:
        result = predictor.predict(request.city, request.date)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.get("/health")
async def health_check():
    return {"status": "ok", "model_loaded": True}
