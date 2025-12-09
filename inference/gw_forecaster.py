"""
Groundwater Level Forecasting - Production Inference Module

Usage:
    from gw_forecaster import GroundwaterForecaster

    forecaster = GroundwaterForecaster()
    prediction = forecaster.predict(station_id=338, gw_history=[10.5, 10.4, 10.3, 10.2, 10.1, 10.0, 9.9])
    print(f"30-day forecast: {prediction:.2f} meters")
"""

import torch
import torch.nn as nn
import numpy as np
import json
import pickle
from sklearn.preprocessing import StandardScaler


class EmbeddingLSTM(nn.Module):
    """LSTM with station ID embeddings for multi-well forecasting."""

    def __init__(self, input_size=1, hidden_size=64, embedding_dim=8, n_stations=392, output_size=1):
        super().__init__()
        self.embedding = nn.Embedding(n_stations, embedding_dim)
        self.lstm = nn.LSTM(input_size + embedding_dim, hidden_size, batch_first=True)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Linear(64, output_size)
        )

    def forward(self, x, station_ids):
        embeddings = self.embedding(station_ids).unsqueeze(1).expand(-1, x.size(1), -1)
        x_combined = torch.cat([x, embeddings], dim=2)
        lstm_out, _ = self.lstm(x_combined)
        last_out = lstm_out[:, -1, :]
        out = self.fc(last_out)
        return out


class GroundwaterForecaster:
    """Production inference wrapper for groundwater forecasting."""

    def __init__(self, model_path='model_production.pth', 
                 scalers_path='scalers_production.pkl',
                 station_map_path='station_mapping.json',
                 device='cuda' if torch.cuda.is_available() else 'cpu'):
        """Initialize forecaster by loading artifacts."""
        self.device = device

        # Load metadata
        with open('metadata.json') as f:
            self.metadata = json.load(f)

        # Load model
        config = self.metadata['config']
        self.model = EmbeddingLSTM(
            input_size=config['input_size'],
            hidden_size=config['hidden_size'],
            embedding_dim=config['embedding_dim'],
            n_stations=config['n_stations'],
            output_size=config['output_size']
        ).to(device)

        model_state = torch.load(model_path, map_location=device, weights_only=True)
        self.model.load_state_dict(model_state)
        self.model.eval()

        # Load scalers
        with open(scalers_path, 'rb') as f:
            scalers_data = pickle.load(f)

        self.scalers = {}
        for station_id, scaler_dict in scalers_data.items():
            scaler_x = StandardScaler()
            scaler_x.mean_ = np.array(scaler_dict['scaler_x_mean'])
            scaler_x.scale_ = np.array(scaler_dict['scaler_x_scale'])

            scaler_y = StandardScaler()
            scaler_y.mean_ = np.array(scaler_dict['scaler_y_mean'])
            scaler_y.scale_ = np.array(scaler_dict['scaler_y_scale'])

            self.scalers[int(station_id)] = {'x': scaler_x, 'y': scaler_y}

        # Load station mapping
        with open(station_map_path) as f:
            self.station_mapping = {int(k): v for k, v in json.load(f).items()}

    def predict(self, station_id, gw_history):
        """
        Make a 30-day groundwater forecast.

        Args:
            station_id (int): Station ID
            gw_history (array-like): Last 7 days of GW levels (meters)

        Returns:
            float: Predicted GW level 30 days from now (meters)
        """
        if station_id not in self.station_mapping:
            raise ValueError(f"Station {station_id} not in model")

        gw_history = np.array(gw_history, dtype=np.float32)
        if len(gw_history) != 7:
            raise ValueError(f"Expected 7 days, got {len(gw_history)}")

        # Scale and predict
        scaler_x = self.scalers[station_id]['x']
        gw_scaled = scaler_x.transform(gw_history.reshape(-1, 1))

        X_input = torch.tensor(gw_scaled, dtype=torch.float32).unsqueeze(0).to(self.device)
        station_idx = self.station_mapping[station_id]
        station_tensor = torch.tensor([station_idx], dtype=torch.long).to(self.device)

        with torch.no_grad():
            y_pred_scaled = self.model(X_input, station_tensor).cpu().numpy()

        scaler_y = self.scalers[station_id]['y']
        y_pred = scaler_y.inverse_transform(y_pred_scaled)[0, 0]

        return float(y_pred)

    def batch_predict(self, predictions_list):
        """Make multiple forecasts efficiently."""
        return [self.predict(item['station_id'], item['gw_history']) 
                for item in predictions_list]
