from pydantic import BaseModel

class SoilInputSchema(BaseModel):
    N: float
    P: float
    K: float
    temperature: float
    humidity: float
    ph: float
    rainfall: float

class PredictionResponseSchema(BaseModel):
    top1: str
    top3: list
    confidence: float