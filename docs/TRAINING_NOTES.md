# Training Notes - Groundwater Forecasting Model

## Key Findings

### 1. Problem Formulation
The original approach failed catastrophically because the prediction horizon was too short:
- 1-day horizon: Lag-1 ACF = 0.98 (changes only 0.01m/day)
- 7-day horizon: Lag-7 ACF = 0.90 (persistence nearly optimal)
- **30-day horizon: Lag-30 ACF = 0.60 (sweet spot for learning)**

### 2. Model Architecture
- Simple EmbeddingLSTM outperformed complex attention models
- Station embeddings (8D) capture well-specific hydrology
- Shared LSTM learns general groundwater physics
- Small model (24.5K params) prevents overfitting

### 3. Training Strategy
- Per-well StandardScaler (not global) essential for numerical stability
- Mini-batch training (batch_size=256) enables 6GB GPU fit
- Early stopping (patience=5) prevents overfitting
- Gradient clipping (norm=1.0) stabilizes LSTM gradients

### 4. Data Quality
- Quality filter: wells with <100 days excluded
- All 392 wells passed quality threshold
- 389K sequences generated (7-day input → 30-day target)
- Train/val/test: 70/15/15 global timeline split

### 5. Results Interpretation
- +27.5% aggregate skill on test set
- Success rate ~88% of wells
- 45 wells still below persistence (domain-specific challenges)
- Model learned well despite heterogeneity

## Production Deployment Checklist

- [ ] Copy production_deployment/ to production server
- [ ] Install requirements: `pip install -r inference/requirements.txt`
- [ ] Test inference: Run gw_forecaster on sample well
- [ ] Setup monitoring: Track forecast vs actual accuracy
- [ ] Configure API: Deploy with Flask/FastAPI
- [ ] Document endpoints: Update team on API structure
- [ ] Backup: Keep monthly snapshots of model
- [ ] Monitoring: Alert if skill drops below -10%

## Future Improvements

1. **Ensemble methods**: Combine with other models for confidence intervals
2. **Seasonal tuning**: Separate models for monsoon vs dry seasons
3. **Uncertainty quantification**: Bayesian LSTM for confidence bounds
4. **Transfer learning**: Pre-train on all India data, fine-tune for Tamil Nadu
5. **Real-time updates**: Online learning with new observations
6. **Domain adaptation**: Handle distribution shift gracefully

## Troubleshooting

### Low accuracy on specific well
- Check: Is well historically different (different geology)?
- Action: Consider well-specific fine-tuning
- Monitor: Compare to persistence baseline for that well

### Inference latency high
- Check: GPU memory pressure
- Action: Reduce batch size or use CPU inference for non-critical forecasts
- Optimize: Profile inference time per component

### Model drift over time
- Check: Monthly retrain on latest 2 years data
- Action: Trigger retraining if validation skill drops >5%
- Monitor: Keep validation set separate for drift detection

---
Last Updated: 2025-12-06
Model Version: 1.0
Status: Production Ready
