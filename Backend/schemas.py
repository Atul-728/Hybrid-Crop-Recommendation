from pydantic import BaseModel


class SoilInputSchema(BaseModel):
    N: float
    P: float
    K: float
    temperature: float
    humidity: float
    ph: float
    rainfall: float
    region: str
    market_price: float = 0
    production_cost: float = 0


class PredictionResponseSchema(BaseModel):
    top1: str
    top3: list
    confidence: float
    risk_score: float
    risk_level: str
    expected_profit: float
