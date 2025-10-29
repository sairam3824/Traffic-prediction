# 🚗 Traffic Congestion Prediction System

A comprehensive AI-powered traffic prediction platform that combines advanced machine learning models with real-time visualization and route planning capabilities.

## 🌟 Features

### 🤖 Advanced AI Models
- **LSTM-CNN-GRU Ensemble**: Hybrid deep learning architecture for accurate traffic prediction
- **Spatio-temporal Analysis**: Graph Neural Networks (GNN) for location-based predictions
- **Real-time Processing**: Sub-second prediction response times
- **High Accuracy**: RMSE ≤ 1.12 with R² > 0.92 performance metrics

### 🗺️ Interactive Dashboard
- **Live Traffic Visualization**: Real-time traffic overlay on Google Maps
- **Route Planning**: Multi-waypoint route optimization with traffic awareness
- **Predictive Analytics**: Next-week traffic forecasting with confidence intervals
- **Performance Monitoring**: Real-time model metrics and system health

### 🔧 Technical Stack
- **Frontend**: Next.js 16, React 19, TypeScript, Tailwind CSS
- **Backend**: Flask API, TensorFlow, scikit-learn
- **Database**: Supabase (PostgreSQL)
- **Maps**: Google Maps API with traffic layer
- **UI Components**: Radix UI, Lucide React icons

## 🚀 Quick Start

### Prerequisites
- Node.js 18+ and npm/pnpm
- Python 3.9+ with pip
- Google Maps API key
- Supabase account (optional)

### 1. Clone and Install
```bash
git clone <repository-url>
cd traffic-cogestion-1

# Install frontend dependencies
npm install

# Install backend dependencies
cd UCS_Model-main
pip install -r requirements_web.txt
```

### 2. Environment Setup
Create `.env.local` in the project root:
```env
# Google Maps API
GOOGLE_MAPS_API_KEY=your_google_maps_api_key

# Flask Backend
FLASK_API_URL=http://localhost:5000

# Supabase (Optional)
NEXT_PUBLIC_SUPABASE_URL=your_supabase_url
NEXT_PUBLIC_SUPABASE_ANON_KEY=your_supabase_anon_key
```

### 3. Start Services

**Terminal 1 - Backend API:**
```bash
cd UCS_Model-main
python traffic_prediction_api.py
```

**Terminal 2 - Frontend:**
```bash
npm run dev
```

### 4. Access the Application
- **Main Dashboard**: http://localhost:3000
- **API Documentation**: http://localhost:5000
- **Health Check**: http://localhost:5000/api/health

## 📊 Application Structure

```
traffic-cogestion-1/
├── app/                          # Next.js App Router
│   ├── traffic-prediction/       # Main dashboard
│   ├── route-planner/           # Route optimization
│   ├── monitoring/              # System monitoring
│   ├── admin/                   # Admin panel
│   └── api/                     # API routes
├── UCS_Model-main/              # ML Backend
│   ├── models/                  # Trained models
│   ├── traffic_prediction_api.py # Flask API
│   └── Capstone_LSTM_CNN-GRU_Notebook.ipynb
├── components/                  # React components
├── lib/                        # Utilities
└── public/                     # Static assets
```

## 🔌 API Endpoints

### Traffic Prediction
```bash
POST /api/ucs-predict
{
  "latitude": 40.7128,
  "longitude": -74.0060,
  "timestamp": "2024-03-15T10:30:00Z"
}
```

### Route Analysis
```bash
POST /api/ucs-predict-route
{
  "waypoints": [
    {"latitude": 40.7128, "longitude": -74.0060},
    {"latitude": 40.7589, "longitude": -73.9851}
  ]
}
```

### Model Information
```bash
GET /api/ucs-model-info
```

## 🧠 Machine Learning Models

### LSTM-CNN-GRU Ensemble
- **Architecture**: Hybrid deep learning combining temporal and spatial features
- **Input Features**: 32 engineered features including location, time, weather
- **Sequence Length**: 10 time steps for temporal pattern recognition
- **Performance**: RMSE 0.0523, MAE 0.0412, R² 0.9234

### Graph Neural Network (GNN)
- **Purpose**: Spatio-temporal traffic flow modeling
- **Technology**: PyTorch Geometric with K-NN graph construction
- **Features**: Zone clustering, spatial relationship modeling
- **Use Case**: City-wide traffic pattern analysis

## 🎯 Key Features

### 📍 Location-Based Predictions
- GPS coordinate input with automatic location detection
- Historical traffic pattern analysis for specific locations
- Point-of-interest (POI) influence modeling
- Real-time traffic condition integration

### 🛣️ Intelligent Route Planning
- Multi-waypoint route optimization
- Traffic-aware path selection
- Dynamic rerouting based on predictions
- ETA calculations with confidence intervals

### 📈 Advanced Analytics
- Next-week traffic forecasting
- Uncertainty quantification with confidence bands
- Anomaly detection for unusual traffic patterns
- Performance metrics dashboard

### 🔄 Real-Time Processing
- WebSocket integration for live updates
- Sub-second prediction response times
- Automatic model retraining pipeline
- Push notifications for traffic alerts

## 🛠️ Development

### Running Tests
```bash
# Frontend tests
npm test

# Backend tests
cd UCS_Model-main
python test_flask_api.py
python test_model_loading.py
```

### Model Training
```bash
cd UCS_Model-main
jupyter notebook Capstone_LSTM_CNN-GRU_Notebook.ipynb
# Run all cells to train and save models
```

### Building for Production
```bash
# Frontend build
npm run build
npm start

# Backend production server
cd UCS_Model-main
gunicorn -w 4 -b 0.0.0.0:5000 traffic_prediction_api:app
```

## 🚀 Deployment

### Vercel (Frontend)
```bash
npm install -g vercel
vercel deploy --prod
```

### Heroku (Backend)
```bash
# Create Procfile
echo "web: gunicorn traffic_prediction_api:app" > UCS_Model-main/Procfile

# Deploy
cd UCS_Model-main
heroku create your-app-name
git push heroku main
```

### Docker
```dockerfile
# Frontend
FROM node:18-alpine
WORKDIR /app
COPY package*.json ./
RUN npm install
COPY . .
RUN npm run build
EXPOSE 3000
CMD ["npm", "start"]

# Backend
FROM python:3.9-slim
WORKDIR /app
COPY UCS_Model-main/requirements_web.txt .
RUN pip install -r requirements_web.txt
COPY UCS_Model-main/ .
EXPOSE 5000
CMD ["python", "traffic_prediction_api.py"]
```

## 📊 Performance Metrics

### Model Performance
- **RMSE**: 0.0523 (Root Mean Square Error)
- **MAE**: 0.0412 (Mean Absolute Error)
- **R²**: 0.9234 (Coefficient of Determination)
- **Prediction Speed**: <100ms per request

### System Performance
- **API Response Time**: <200ms average
- **Frontend Load Time**: <2s initial load
- **Concurrent Users**: 100+ supported
- **Uptime**: 99.9% target availability

## 🔧 Configuration

### Google Maps Setup
1. Enable APIs in Google Cloud Console:
   - Maps JavaScript API
   - Directions API
   - Places API
   - Geocoding API

2. Add API key to environment variables
3. Configure billing and usage limits

### Supabase Setup (Optional)
1. Create new Supabase project
2. Run SQL scripts from `CREATE_SUPABASE_TABLE.md`
3. Configure Row Level Security (RLS)
4. Add connection details to environment

## 🐛 Troubleshooting

### Common Issues

**Model Loading Errors**
```bash
# Check model files exist
ls -la UCS_Model-main/models/
# Should contain: traffic_prediction_model.h5, feature_scaler.pkl, target_scaler.pkl
```

**Port Conflicts**
```bash
# Kill process on port 5000
lsof -ti:5000 | xargs kill -9
# Or change port in traffic_prediction_api.py
```

**Google Maps Not Loading**
- Verify API key is correct
- Check API quotas and billing
- Ensure required APIs are enabled

### Debug Tools
- **Debug Console**: http://localhost:3000/debug-console.html
- **Test Interface**: http://localhost:3000/test-map.html
- **API Health**: http://localhost:5000/api/health

## 📚 Documentation

- [Complete Setup Guide](COMPLETE_SETUP_GUIDE.md)
- [Deployment Instructions](DEPLOYMENT.md)
- [Supabase Integration](CREATE_SUPABASE_TABLE.md)
- [Testing Guide](TEST_SUPABASE_INTEGRATION.md)
- [Troubleshooting](TROUBLESHOOTING.md)

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

## 📄 License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- TensorFlow team for deep learning framework
- Google Maps Platform for mapping services
- Supabase for database infrastructure
- Next.js team for React framework
- Open source community for various libraries

## 📞 Support

For support and questions:
- Create an issue in the repository
- Check the troubleshooting guide
- Review the complete setup documentation

---
