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
    market_price: float
    production_cost: float

class PredictionResponseSchema(BaseModel):
    top1: str
    top3: list
    confidence: float
    risk_score: float
    expected_profit: float