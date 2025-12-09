---
title: Groundwater Forecaster API
emoji: 💧
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
app_port: 7860
---

# Groundwater Level Forecasting API

FastAPI service for 30-day groundwater level predictions using ensemble ML model.

## API Usage

### Health Check
```bash
GET https://YOUR-SPACE-NAME.hf.space/ping
```

### Forecast Endpoint
```bash
POST https://YOUR-SPACE-NAME.hf.space/api/v1/forecast

{
  "station_id": 1001,
  "gw_history_7days": [10.5, 10.3, 10.2, 10.1, 10.0, 9.9, 9.8],
  "rainfall_30d": 50.0,
  "temp_mean": 25.0,
  "temp_range": 10.0,
  "pet_7d": 35.0
}
```

## React Native Integration

```javascript
const API_URL = 'https://YOUR-SPACE-NAME.hf.space';

const getForecast = async (stationId, gwHistory) => {
  const response = await fetch(`${API_URL}/api/v1/forecast`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      station_id: stationId,
      gw_history_7days: gwHistory,
      rainfall_30d: 50.0,
      temp_mean: 25.0,
      temp_range: 10.0,
      pet_7d: 35.0
    })
  });
  return await response.json();
};
```
