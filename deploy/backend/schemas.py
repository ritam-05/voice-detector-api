from pydantic import BaseModel

class PredictResponse(BaseModel):
    label: str
    score: float
    reason: str
