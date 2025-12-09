// React Native API Service for Groundwater Forecasting

const HUGGINGFACE_API_URL = 'https://YOUR-SPACE-NAME.hf.space';

export const GroundwaterAPI = {
  // Health check
  async ping() {
    try {
      const response = await fetch(`${HUGGINGFACE_API_URL}/ping`);
      return await response.json();
    } catch (error) {
      console.error('Ping failed:', error);
      throw error;
    }
  },

  // Get 30-day forecast
  async getForecast(params) {
    try {
      const response = await fetch(`${HUGGINGFACE_API_URL}/api/v1/forecast`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          station_id: params.stationId,
          gw_history_7days: params.gwHistory,
          rainfall_30d: params.rainfall || null,
          temp_mean: params.tempMean || null,
          temp_range: params.tempRange || null,
          pet_7d: params.pet || null,
        }),
      });

      if (!response.ok) {
        throw new Error(`API Error: ${response.status}`);
      }

      const data = await response.json();
      return data;
    } catch (error) {
      console.error('Forecast request failed:', error);
      throw error;
    }
  },
};

// Example Usage in React Native Component:
/*
import React, { useState } from 'react';
import { View, Button, Text, ActivityIndicator } from 'react-native';
import { GroundwaterAPI } from './services/GroundwaterAPI';

export default function ForecastScreen() {
  const [forecast, setForecast] = useState(null);
  const [loading, setLoading] = useState(false);

  const handleGetForecast = async () => {
    setLoading(true);
    try {
      const result = await GroundwaterAPI.getForecast({
        stationId: 1001,
        gwHistory: [10.5, 10.3, 10.2, 10.1, 10.0, 9.9, 9.8],
        rainfall: 50.0,
        tempMean: 25.0,
        tempRange: 10.0,
        pet: 35.0,
      });
      
      setForecast(result);
      console.log('Forecast:', result.forecast_30day);
      console.log('Mean:', result.forecast_mean);
      console.log('Confidence:', result.confidence);
    } catch (error) {
      console.error('Error:', error);
    } finally {
      setLoading(false);
    }
  };

  return (
    <View>
      <Button title="Get Forecast" onPress={handleGetForecast} />
      {loading && <ActivityIndicator />}
      {forecast && (
        <View>
          <Text>Mean Forecast: {forecast.forecast_mean.toFixed(2)}m</Text>
          <Text>Confidence: {(forecast.confidence * 100).toFixed(1)}%</Text>
          <Text>30-Day Trend: {forecast.forecast_30day.join(', ')}</Text>
        </View>
      )}
    </View>
  );
}
*/
