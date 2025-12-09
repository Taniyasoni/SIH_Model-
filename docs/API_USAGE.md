# API Usage Guide - Groundwater Forecasting Model

## Quick Start

```python
from inference.gw_forecaster import GroundwaterForecaster

# Initialize
forecaster = GroundwaterForecaster()

# Make prediction
prediction = forecaster.predict(
    station_id=338,
    gw_history=[10.5, 10.4, 10.3, 10.2, 10.1, 10.0, 9.9]
)
print(f"30-day forecast: {prediction:.2f} meters")
```

## REST API Example

```bash
curl -X POST http://localhost:5000/api/forecast \
  -H "Content-Type: application/json" \
  -d '{
    "station_id": 338,
    "gw_history": [10.5, 10.4, 10.3, 10.2, 10.1, 10.0, 9.9]
  }'

# Response
{
  "station_id": 338,
  "current_gw": 9.9,
  "forecast_30d": -6.06,
  "change_meters": -0.56,
  "confidence": 0.68
}
```

## Batch Prediction

```python
# Multiple forecasts at once
batch = [
    {"station_id": 338, "gw_history": [...7 days...]},
    {"station_id": 328, "gw_history": [...7 days...]},
    {"station_id": 335, "gw_history": [...7 days...]},
]

forecasts = forecaster.batch_predict(batch)
# Returns: [forecast1, forecast2, forecast3]
```

## Requirements

- PyTorch 2.0+
- NumPy
- scikit-learn
- Python 3.8+

See `requirements.txt` for exact versions.
