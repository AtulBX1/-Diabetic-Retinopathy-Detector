from pydantic import BaseModel


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool


class PredictionResponse(BaseModel):
    grade: int
    label: str
    confidence: float
    probabilities: list[float]
    low_confidence: bool
    threshold: float
    gradcam_base64: str


class ErrorResponse(BaseModel):
    detail: str
