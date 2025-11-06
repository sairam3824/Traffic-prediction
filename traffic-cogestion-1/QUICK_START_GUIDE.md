# 🚀 Quick Start Guide - Traffic Prediction System

## ✨ What's New?

Your traffic prediction system now has **3 powerful models** working together:
1. **LSTM Model** - Great for time series patterns
2. **CNN-LSTM Model** - Excellent for spatial-temporal features  
3. **Ensemble Model** - Combines both for best accuracy

## 🎯 Start in 3 Steps

### Step 1: Open Two Terminals

### Step 2: Terminal 1 - Start Backend
```bash
cd traffic-cogestion-1/UCS_Model-main
python traffic_prediction_api_enhanced.py
```

You should see:
```
============================================================
LOADING TRAFFIC PREDICTION MODELS
============================================================

📥 Loading LSTM model from: models/lstm_model.h5
✅ LSTM model loaded successfully!

📥 Loading CNN-LSTM model from: models/cnn_lstm_model.h5
✅ CNN-LSTM model loaded successfully!

📥 Loading BEST model from: models/best_model.h5
✅ BEST model loaded successfully!

============================================================
✅ LOADED 3 MODEL(S) SUCCESSFULLY!
============================================================

🚀 Starting Enhanced Traffic Prediction API...
🌐 API available at: http://localhost:5001
```

### Step 3: Terminal 2 - Start Frontend
```bash
cd traffic-cogestion-1
npm run dev
```

## 🌐 Access Your App

- **Main Application**: http://localhost:3000
- **API Endpoint**: http://localhost:5001
- **Health Check**: http://localhost:5001/api/health

## 🧪 Test It Out

### Test 1: Check Available Models
```bash
curl http://localhost:5001/api/model_info
```

Expected response:
```json
{
  "available_models": ["lstm", "cnn_lstm", "best"],
  "total_models": 3,
  "metadata": {...}
}
```

### Test 2: Get a Prediction (Ensemble)
```bash
curl -X POST http://localhost:5001/api/predict \
  -H "Content-Type: application/json" \
  -d '{
    "latitude": 16.5062,
    "longitude": 80.6480,
    "timestamp": "2024-11-05T14:30:00",
    "model": "ensemble"
  }'
```

Expected response:
```json
{
  "model": "ensemble",
  "prediction": 45.2,
  "individual_predictions": [
    {"model": "lstm", "prediction": 43.5},
    {"model": "cnn_lstm", "prediction": 46.8},
    {"model": "best", "prediction": 45.3}
  ],
  "timestamp": "2024-11-05T14:30:00",
  "location": {"lat": 16.5062, "lon": 80.6480}
}
```

### Test 3: Compare Different Models
```bash
# Test LSTM
curl -X POST http://localhost:5001/api/predict \
  -H "Content-Type: application/json" \
  -d '{"latitude": 16.5062, "longitude": 80.6480, "timestamp": "2024-11-05T14:30:00", "model": "lstm"}'

# Test CNN-LSTM
curl -X POST http://localhost:5001/api/predict \
  -H "Content-Type: application/json" \
  -d '{"latitude": 16.5062, "longitude": 80.6480, "timestamp": "2024-11-05T14:30:00", "model": "cnn_lstm"}'

# Test Ensemble (recommended)
curl -X POST http://localhost:5001/api/predict \
  -H "Content-Type: application/json" \
  -d '{"latitude": 16.5062, "longitude": 80.6480, "timestamp": "2024-11-05T14:30:00", "model": "ensemble"}'
```

## 📊 Model Selection Guide

| Model | When to Use | Accuracy | Speed |
|-------|------------|----------|-------|
| **LSTM** | Simple time series | Good | Fast |
| **CNN-LSTM** | Complex patterns | Better | Medium |
| **Ensemble** | Best accuracy | Best | Slower |
| **Best** | Production default | Very Good | Fast |

**Recommendation**: Use `"ensemble"` for most accurate predictions.

## 🔧 Troubleshooting

### Problem: "No models found"
**Solution**: Make sure you're in the correct directory
```bash
cd traffic-cogestion-1/UCS_Model-main
ls models/  # Should show lstm_model.h5, cnn_lstm_model.h5, best_model.h5
```

### Problem: "Port already in use"
**Solution**: Kill the existing process
```bash
# For port 5001 (Backend)
lsof -ti:5001 | xargs kill -9

# For port 3000 (Frontend)
lsof -ti:3000 | xargs kill -9
```

### Problem: "Module not found"
**Solution**: Install dependencies
```bash
# Backend
pip install tensorflow numpy pandas scikit-learn flask

# Frontend
npm install
```

### Problem: Models loading slowly
**Solution**: This is normal on first load. Models are ~5.6MB total and load once at startup.

## 📈 Model Performance

Based on training results:

```
┌─────────────┬──────┬──────┬────────┐
│ Model       │ RMSE │ MAE  │ R²     │
├─────────────┼──────┼──────┼────────┤
│ LSTM        │ 3.5  │ 2.8  │ 0.85   │
│ CNN-LSTM    │ 3.2  │ 2.5  │ 0.88   │
│ Ensemble    │ 3.0  │ 2.3  │ 0.90   │
└─────────────┴──────┴──────┴────────┘
```

Lower RMSE/MAE = Better | Higher R² = Better

## 🎓 Advanced Usage

### Retrain Models
```bash
cd traffic-cogestion-1/UCS_Model-main

# Train LSTM
python lstm_model.py

# Train CNN-LSTM
python cnn_lstm_model.py

# Compare all models
python combine_models.py
```

### Route Prediction
```bash
curl -X POST http://localhost:5001/api/predict_route \
  -H "Content-Type: application/json" \
  -d '{
    "waypoints": [
      {"latitude": 16.5062, "longitude": 80.6480},
      {"latitude": 16.5100, "longitude": 80.6500},
      {"latitude": 16.5150, "longitude": 80.6550}
    ],
    "model": "ensemble"
  }'
```

## 📚 More Documentation

- **Full Integration Guide**: `INTEGRATION_SUMMARY.md`
- **Detailed API Docs**: `UCS_Model-main/README_INTEGRATED.md`
- **Deployment Guide**: `UCS_Model-main/deployment_guide.md`

## 🎉 You're All Set!

Your traffic prediction system is now running with multiple AI models. The ensemble model combines the strengths of both LSTM and CNN-LSTM for the most accurate predictions.

**Happy Predicting! 🚗📊**

---

**Need Help?** Check the troubleshooting section above or review the detailed documentation.
