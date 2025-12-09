"""
Ensemble Groundwater Forecaster
================================

Combines 3 models for robust 30-day groundwater level forecasting:
1. Persistence model (50%): Today's GW = forecast
2. LSTM model (30%): Learned neural network patterns
3. Rainfall model (20%): Weather-driven GW change

Usage:
------
from gw_ensemble_forecaster import GroundwaterEnsembleForecaster

forecaster = GroundwaterEnsembleForecaster()

# Single well forecast
forecast_30day = forecaster.predict(
    station_id=338,
    gw_history_7days=np.array([10.2, 10.1, 10.0, 9.9, 9.8, 9.7, 9.6]),
    rainfall_30d=120.5,
    temp_mean=28.3,
    temp_range=8.1,
    pet_7d=35.2
)

# Batch forecast
batch_forecast = forecaster.batch_predict(requests=[...])
"""

import numpy as np
import torch
import torch.nn as nn
import pickle
import json
import warnings
from pathlib import Path
from typing import List, Dict, Optional, Tuple

warnings.filterwarnings('ignore')


class EmbeddingLSTM(nn.Module):
    """LSTM with station embeddings for multi-well forecasting"""
    def __init__(self, num_stations: int, embedding_dim: int = 8, 
                 hidden_dim: int = 64):
        super().__init__()
        self.num_stations = num_stations
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        
        # Station embedding layer
        self.station_embedding = nn.Embedding(num_stations, embedding_dim)
        
        # LSTM layer (single layer, as per saved model)
        input_size = 1 + embedding_dim  # GW level + station embedding
        self.lstm = nn.LSTM(input_size, hidden_dim, batch_first=True)
        
        # Output layers (as per saved model: 64 -> 32 -> 1)
        self.fc1 = nn.Linear(hidden_dim, 32)
        self.fc2 = nn.Linear(32, 1)
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, gw_history, station_ids):
        """
        Args:
            gw_history: (batch, seq_len, 1)
            station_ids: (batch,)
        Returns:
            forecast: (batch, 1) - next day GW level
        """
        batch_size = gw_history.size(0)
        
        # Get embeddings
        embeddings = self.station_embedding(station_ids)  # (batch, embedding_dim)
        embeddings = embeddings.unsqueeze(1).expand(-1, gw_history.size(1), -1)  # (batch, seq, embedding_dim)
        
        # Concatenate GW history with embeddings
        x = torch.cat([gw_history, embeddings], dim=2)  # (batch, seq, 1+embedding_dim)
        
        # LSTM layer
        x, _ = self.lstm(x)
        
        # Take last output
        x = x[:, -1, :]  # (batch, hidden_dim)
        
        # FC layers
        x = self.fc1(x)
        x = self.dropout(x)
        forecast = self.fc2(x)  # (batch, 1)
        return forecast


class GroundwaterEnsembleForecaster:
    """Production-ready ensemble forecaster for 30-day GW predictions"""
    
    def __init__(self, production_dir: str = 'production_deployment'):
        """
        Initialize ensemble forecaster from production artifacts
        
        Args:
            production_dir: Path to production_deployment folder
        """
        self.production_dir = Path(production_dir)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load LSTM model
        self._load_lstm_model()
        
        # Load rainfall Ridge model and scaler
        self._load_rainfall_model()
        
        # Load station mapping and scalers
        self._load_mappings()
        
        print(f"✅ Ensemble forecaster initialized")
        print(f"   Device: {self.device}")
        print(f"   LSTM model: EmbeddingLSTM (24,457 params)")
        print(f"   Rainfall model: Ridge regression")
        print(f"   Ensemble weights: 50% persistence, 30% LSTM, 20% rainfall")
    
    def _load_lstm_model(self):
        """Load trained LSTM model"""
        model_path = self.production_dir / 'model' / 'model_production.pth'
        if not model_path.exists():
            raise FileNotFoundError(f"LSTM model not found at {model_path}")
        
        # Instantiate model (matches training config)
        # Note: 393 stations to match saved checkpoint (index 0-392)
        self.lstm_model = EmbeddingLSTM(
            num_stations=393,
            embedding_dim=8,
            hidden_dim=64
        )
        
        # Load weights
        checkpoint = torch.load(model_path, map_location=self.device)
        self.lstm_model.load_state_dict(checkpoint)
        self.lstm_model.to(self.device)
        self.lstm_model.eval()
        
        print("   ✓ LSTM model loaded")
    
    def _load_rainfall_model(self):
        """Load rainfall Ridge model and feature scaler"""
        rainfall_model_path = self.production_dir / 'model' / 'rainfall_model.pkl'
        scaler_path = self.production_dir / 'model' / 'rainfall_scaler.pkl'
        
        if not rainfall_model_path.exists() or not scaler_path.exists():
            print(f"   ⚠ Rainfall model not found, will use only LSTM + persistence")
            self.rainfall_model = None
            self.rainfall_scaler = None
            return
        
        with open(rainfall_model_path, 'rb') as f:
            self.rainfall_model = pickle.load(f)
        
        with open(scaler_path, 'rb') as f:
            self.rainfall_scaler = pickle.load(f)
        
        print("   ✓ Rainfall model loaded")
    
    def _load_mappings(self):
        """Load station mapping and scalers"""
        mapping_path = self.production_dir / 'mappings' / 'station_mapping.json'
        
        if not mapping_path.exists():
            raise FileNotFoundError(f"Station mapping not found at {mapping_path}")
        
        with open(mapping_path, 'r') as f:
            self.station_to_idx = json.load(f)
        
        # Load per-well scalers
        scalers_path = self.production_dir / 'scalers' / 'scalers_production.pkl'
        if scalers_path.exists():
            with open(scalers_path, 'rb') as f:
                self.well_scalers = pickle.load(f)
        else:
            self.well_scalers = {}
        
        print(f"   ✓ Loaded {len(self.station_to_idx)} station mappings")
    
    def predict(self, 
                station_id: int,
                gw_history_7days: np.ndarray,
                rainfall_30d: float = None,
                temp_mean: float = None,
                temp_range: float = None,
                pet_7d: float = None) -> Dict:
        """
        Generate 30-day GW forecast for a single well using ensemble
        
        Args:
            station_id: Well station ID
            gw_history_7days: Last 7 days GW levels (meters)
            rainfall_30d: 30-day cumulative rainfall (mm) - optional
            temp_mean: Mean temperature (°C) - optional
            temp_range: Temperature range (°C) - optional
            pet_7d: 7-day PET (mm) - optional
        
        Returns:
            dict with keys:
                - forecast_30day: (30,) array of daily forecasts
                - forecast_mean: Mean of 30-day forecast
                - forecast_std: Std of 30-day forecast
                - components: dict with individual model forecasts
                - weights: dict with ensemble weights
                - confidence: Confidence score (0-1)
        """
        # Validate inputs
        station_key = str(station_id)
        if station_key not in self.station_to_idx:
            raise ValueError(f"Station {station_id} not in training set (392 stations)")
        
        gw_history_7days = np.array(gw_history_7days, dtype=np.float32)
        if len(gw_history_7days) != 7:
            raise ValueError(f"Expected 7 days of history, got {len(gw_history_7days)}")
        
        # Get station index
        station_idx = self.station_to_idx[station_key]
        
        # MODEL 1: Persistence (today = forecast)
        forecast_persistence = np.full(30, gw_history_7days[-1])
        
        # MODEL 2: LSTM inference
        forecast_lstm = self._lstm_predict(gw_history_7days, station_idx)
        
        # MODEL 3: Rainfall model (if available)
        if self.rainfall_model is not None and all(x is not None for x in 
                                                    [rainfall_30d, temp_mean, temp_range, pet_7d]):
            forecast_rainfall = self._rainfall_predict(
                gw_history_7days[-1],  # Today's GW
                rainfall_30d, temp_mean, temp_range, pet_7d
            )
        else:
            forecast_rainfall = forecast_persistence  # Fallback
        
        # ENSEMBLE: Weighted blend
        weights = {'persistence': 0.50, 'lstm': 0.30, 'rainfall': 0.20}
        forecast_ensemble = (
            weights['persistence'] * forecast_persistence +
            weights['lstm'] * forecast_lstm +
            weights['rainfall'] * forecast_rainfall
        )
        
        return {
            'station_id': station_id,
            'forecast_30day': forecast_ensemble.tolist(),
            'forecast_mean': float(forecast_ensemble.mean()),
            'forecast_std': float(forecast_ensemble.std()),
            'components': {
                'persistence': forecast_persistence.tolist(),
                'lstm': forecast_lstm.tolist(),
                'rainfall': forecast_rainfall.tolist() if self.rainfall_model else None
            },
            'weights': weights,
            'confidence': float(self._compute_confidence(forecast_ensemble))
        }
    
    def _lstm_predict(self, gw_history: np.ndarray, station_idx: int) -> np.ndarray:
        """Run LSTM inference (iterative day-by-day for 30 days)"""
        forecast_30day = []
        current_history = gw_history.copy()
        
        with torch.no_grad():
            for day in range(30):
                # Prepare input
                gw_tensor = torch.from_numpy(current_history.reshape(1, 7, 1)).float().to(self.device)
                station_tensor = torch.tensor([station_idx], dtype=torch.long).to(self.device)
                
                # Forward pass
                forecast_next_day = self.lstm_model(gw_tensor, station_tensor)
                forecast_value = forecast_next_day.cpu().numpy()[0, 0]
                forecast_30day.append(forecast_value)
                
                # Update history for next day (shift and add new prediction)
                current_history = np.concatenate([current_history[1:], [forecast_value]])
        
        return np.array(forecast_30day)
    
    def _rainfall_predict(self, 
                         gw_today: float,
                         rainfall_30d: float,
                         temp_mean: float,
                         temp_range: float,
                         pet_7d: float) -> np.ndarray:
        """Ridge regression forecast based on weather"""
        if self.rainfall_model is None or self.rainfall_scaler is None:
            return np.full(30, gw_today)
        
        # Scale features
        X = np.array([[rainfall_30d, temp_mean, temp_range, pet_7d]])
        X_scaled = self.rainfall_scaler.transform(X)
        
        # Predict 30-day change
        gw_change = self.rainfall_model.predict(X_scaled)[0]
        
        # Create 30-day forecast (linear ramp)
        forecast = np.linspace(gw_today, gw_today + gw_change, 30)
        
        return forecast
    
    def _compute_confidence(self, forecast: np.ndarray) -> float:
        """Estimate prediction confidence (0-1)"""
        # Low variance = higher confidence
        std = forecast.std()
        confidence = max(0.0, min(1.0, 1.0 - (std / 10.0)))  # Normalize by typical GW range
        return confidence
    
    def batch_predict(self, requests: List[Dict]) -> List[Dict]:
        """
        Forecast for multiple wells
        
        Args:
            requests: List of dicts with keys:
                - station_id
                - gw_history_7days
                - rainfall_30d (optional)
                - temp_mean (optional)
                - temp_range (optional)
                - pet_7d (optional)
        
        Returns:
            List of forecast dicts (one per request)
        """
        results = []
        for req in requests:
            try:
                result = self.predict(**req)
                results.append(result)
            except Exception as e:
                results.append({'error': str(e), 'station_id': req.get('station_id')})
        
        return results


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == '__main__':
    print("Groundwater Ensemble Forecaster")
    print("=" * 70)
    print()
    
    # Initialize
    forecaster = GroundwaterEnsembleForecaster(
        production_dir='production_deployment'
    )
    
    print()
    print("Example 1: Single well prediction")
    print("-" * 70)
    
    # Example request
    prediction = forecaster.predict(
        station_id=338,
        gw_history_7days=np.array([10.2, 10.1, 10.0, 9.9, 9.8, 9.7, 9.6]),
        rainfall_30d=120.5,
        temp_mean=28.3,
        temp_range=8.1,
        pet_7d=35.2
    )
    
    print(f"Station {prediction['station_id']}:")
    print(f"  Forecast (30-day mean): {prediction['forecast_mean']:.3f}m")
    print(f"  Forecast std dev:       {prediction['forecast_std']:.3f}m")
    print(f"  Confidence:             {prediction['confidence']:.1%}")
    print()
    print("  Component forecasts:")
    for model, forecast in prediction['components'].items():
        if forecast is not None:
            print(f"    {model:15s}: {forecast.mean():.3f}m (mean), "
                  f"{forecast[0]:.3f}m (day 1), {forecast[-1]:.3f}m (day 30)")
    
    print()
    print("✅ Ensemble forecaster ready for production")
