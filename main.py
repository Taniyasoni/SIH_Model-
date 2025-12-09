# main.py  (put this next to docs/, inference/, model/, scalers/)
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np
from pathlib import Path

from inference.gw_ensemble_forecaster import GroundwaterEnsembleForecaster

app = FastAPI()

# Enable CORS for React Native/Web App
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your app's domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/ping")
def ping():
    return {"status": "ok"}

# This class loads:
# - model/model_production.pth
# - model/rainfall_model.pkl
# - scalers/scalers_production.pkl
# - mappings/station_mapping.json
# as described in the guide.
forecaster = GroundwaterEnsembleForecaster(Path(__file__).parent)


class ForecastRequest(BaseModel):
    station_id: int
    gw_history_7days: list[float]  # last 7 days groundwater levels

    # weather is optional – we keep it simple for now
    rainfall_30d: float | None = None
    temp_mean: float | None = None
    temp_range: float | None = None
    pet_7d: float | None = None


class ForecastResponse(BaseModel):
    station_id: int
    forecast_30day: list[float]
    forecast_mean: float
    forecast_std: float
    confidence: float
    components: dict
    weights: dict


@app.post("/api/v1/forecast", response_model=ForecastResponse)
async def get_forecast(req: ForecastRequest):
    try:
        result = forecaster.predict(
            station_id=req.station_id,
            gw_history_7days=np.array(req.gw_history_7days, dtype=float),
            rainfall_30d=req.rainfall_30d,
            temp_mean=req.temp_mean,
            temp_range=req.temp_range,
            pet_7d=req.pet_7d,
        )
        # result already has the correct keys per GUIDE_TECHNICAL
        return ForecastResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
