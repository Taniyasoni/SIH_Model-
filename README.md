# Jal-Chetna ML Model API

Groundwater level forecasting using 3-model ensemble approach.

## Deployment on Render

This repository is ready for Docker deployment on Render.com

## API Endpoints

- `GET /ping` - Health check
- `POST /api/v1/forecast` - Get 30-day groundwater forecast

## Local Development

```bash
# Install dependencies
pip install -r requirements.txt

# Run server
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```
