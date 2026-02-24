# dass_bot/app/main.py
from enum import Enum
from typing import List, Dict, Any
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, validator

from dass_bot.predict import (
    predict_depression,
    predict_anxiety,
    predict_stress,
    predict_all,
)

APP_TITLE = "DASS-42 Mental Health Classifier API"
APP_VERSION = "0.1.0"

app = FastAPI(title=APP_TITLE, version=APP_VERSION)


# ---------- Pydantic schemas ----------
class TargetEnum(str, Enum):
    depression = "depression"
    anxiety = "anxiety"
    stress = "stress"
    all = "all"

class PredictRequest(BaseModel):
    answers: List[int]
    target: TargetEnum = TargetEnum.all
    return_proba: bool = False

    @validator("answers")
    def validate_answers(cls, v):
        if len(v) != 42:
            raise ValueError(f"'answers' must have length 42, got {len(v)}.")
        bad = [x for x in v if x not in (1, 2, 3, 4)]
        if bad:
            raise ValueError(f"All answers must be in [1,2,3,4]. Invalid values: {bad}")
        return v

class PredictResponse(BaseModel):
    result: Dict[str, Any]


# ---------- Routes ----------
@app.get("/", response_class=HTMLResponse, tags=["meta"])
async def root():
    return """
    <h1>Welcome to DASS Prediction API</h1>
    <p>Go to <a href="/docs">/docs</a> to test the endpoints.</p>
    """

@app.get("/health", tags=["meta"])
def health() -> Dict[str, str]:
    return {"status": "ok", "version": APP_VERSION}

@app.post("/predict", response_model=PredictResponse, tags=["inference"])
def predict(req: PredictRequest):
    try:
        if req.target == TargetEnum.depression:
            res = {"depression": predict_depression(req.answers, req.return_proba)}
        elif req.target == TargetEnum.anxiety:
            res = {"anxiety": predict_anxiety(req.answers, req.return_proba)}
        elif req.target == TargetEnum.stress:
            res = {"stress": predict_stress(req.answers, req.return_proba)}
        else:
            res = predict_all(req.answers, req.return_proba)
        return {"result": res}
    except FileNotFoundError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
