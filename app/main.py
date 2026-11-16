from contextlib import asynccontextmanager

import mlflow.sklearn
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel, Field

MODEL_PATH = "model"
state = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    state["model"] = mlflow.sklearn.load_model(MODEL_PATH)
    yield


app = FastAPI(lifespan=lifespan)


class PredictRequest(BaseModel):
    features: list[float] = Field(min_length=64, max_length=64)


@app.post("/predict")
def predict(req: PredictRequest):
    x = np.array(req.features).reshape(1, -1)
    return {"prediction": int(state["model"].predict(x)[0])}