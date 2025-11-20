# 🚦 Traffic Congestion Prediction System

An AI-powered traffic prediction platform combining advanced machine learning models (LSTM-CNN-GRU ensemble, Graph Neural Networks) with real-time visualization, route planning, and predictive analytics.

## 🌟 Overview

This system provides accurate traffic congestion predictions using deep learning models trained on spatio-temporal traffic data. It features a modern Next.js dashboard with Google Maps integration, Flask-based ML API, and comprehensive analytics tools.

### Key Capabilities

- **Real-time Traffic Prediction**: Sub-second predictions with 92%+ accuracy (R² > 0.92)
- **Intelligent Route Planning**: Multi-waypoint optimization with traffic-aware routing
- **Advanced Analytics**: Next-week forecasting, anomaly detection, confidence intervals
- **Interactive Dashboard**: Live traffic visualization, heatmaps, performance monitoring
- **Hybrid ML Architecture**: LSTM-CNN-GRU ensemble + Graph Neural Networks

## 🏗️ Project Structure

```
.
├── traffic/                          # Original ML models and notebooks
│   ├── Cap_UCS_1.ipynb              # Initial model development
│   ├── lstm_model.py                # LSTM implementation
│   ├── cnn_lstm_model.py            # CNN-LSTM hybrid
│   └── final_dataset.csv            # Training data
│
├── traffic-cogestion-1/             # Main application
│   ├── app/                         # Next.js 16 App Router
│   │   ├── traffic-prediction/      # Main dashboard
│   │   ├── route-planner/          # Route optimization
│   │   ├── monitoring/             # System monitoring
│   │   └── api/                    # API routes
│   │
│   ├── UCS_Model-main/             # ML Backend (Flask API)
│   │   ├── models/                 # Trained model files
│   │   ├── traffic_prediction_api.py
│   │   ├── traffic_prediction_api_enhanced.py
│   │   ├── Capstone_LSTM_CNN-GRU_Notebook.ipynb
│   │   └── GNN_PyG_spatio_temporal.py
│   │
│   ├── components/                 # React components
│   │   ├── traffic-history-chart.tsx
│   │   ├── route-map.tsx
│   │   ├── congestion-distribution.tsx
│   │   └── weekly-traffic-heatmap.tsx
│   │
│   └── lib/                       # Utilities and helpers
│
└── README.md                      # This file
```

## 🚀 Quick Start

### Prerequisites

- **Node.js** 18+ and npm/pnpm
- **Python** 3.9+ with pip
- **Google Maps API** key
- **Supabase** account (optional, for data persistence)

### Installation

1. **Clone the repository**
```bash
git clone <repository-url>
cd traffic-congestion-prediction
```

2. **Install frontend dependencies**
```bash
cd traffic-cogestion-1
npm install
```

3. **Install backend dependencies**
```bash
cd traffic-cogestion-1/UCS_Model-main
pip install -r requirements_web.txt
```

### Configuration

Create `.env.local` in `traffic-cogestion-1/`:

```env
# Google Maps API
GOOGLE_MAPS_API_KEY=your_google_maps_api_key

# Flask Backend
FLASK_API_URL=http://localhost:5000

# Supabase (Optional)
NEXT_PUBLIC_SUPABASE_URL=your_supabase_url
NEXT_PUBLIC_SUPABASE_ANON_KEY=your_supabase_anon_key
```

### Running the Application

**Terminal 1 - Start Backend API:**
```bash
cd traffic-cogestion-1/UCS_Model-main
python traffic_prediction_api.py
```

**Terminal 2 - Start Frontend:**
```bash
cd traffic-cogestion-1
npm run dev
```

**Access the application:**
- Dashboard: http://localhost:3000
- API: http://localhost:5000
- Health Check: http://localhost:5000/api/health

## 🤖 Machine Learning Models

### 1. LSTM-CNN-GRU Ensemble

**Architecture:**
- Hybrid deep learning combining temporal and spatial features
- 32 engineered features (location, time, weather, traffic conditions)
- Sequence length: 10 time steps for pattern recognition

**Performance:**
- RMSE: 0.0523
- MAE: 0.0412
- R²: 0.9234
- Prediction Speed: <100ms

**Features:**
- Self-attention mechanism for temporal focus
- Convolutional layers for spatial feature extraction
- GRU units for efficient sequence processing

### 2. Graph Neural Network (GNN)

**Purpose:** Spatio-temporal traffic flow modeling across city zones

**Technology:**
- PyTorch Geometric with K-NN graph construction
- Zone clustering for spatial relationships
- GCN + GRU architecture

**Use Cases:**
- City-wide traffic pattern analysis
- Zone-to-zone traffic flow prediction
- Network-level congestion forecasting

### Model Training

To retrain models with new data:

```bash
cd traffic-cogestion-1/UCS_Model-main

# Open Jupyter notebook
jupyter notebook Capstone_LSTM_CNN-GRU_Notebook.ipynb

# Or run GNN training
python GNN_PyG_spatio_temporal.py
```

## 🔌 API Reference

### Traffic Prediction

```bash
POST /api/ucs-predict
Content-Type: application/json

{
  "latitude": 40.7128,
  "longitude": -74.0060,
  "timestamp": "2024-03-15T10:30:00Z"
}

Response:
{
  "prediction": 65.3,
  "confidence": 0.92,
  "traffic_level": "moderate",
  "model_version": "1.0"
}
```

### Route Analysis

```bash
POST /api/ucs-predict-route
Content-Type: application/json

{
  "waypoints": [
    {"latitude": 40.7128, "longitude": -74.0060},
    {"latitude": 40.7589, "longitude": -73.9851}
  ]
}

Response:
{
  "route_predictions": [...],
  "total_distance": 5.2,
  "estimated_time": 18,
  "congestion_score": 58.7
}
```

### Model Information

```bash
GET /api/ucs-model-info

Response:
{
  "model_type": "LSTM-CNN-GRU",
  "version": "1.0",
  "accuracy_metrics": {
    "rmse": 0.0523,
    "mae": 0.0412,
    "r2": 0.9234
  },
  "features_count": 32,
  "last_trained": "2024-03-01"
}
```

## 🎯 Features

### Dashboard Components

1. **Traffic Prediction Panel**
   - GPS-based location input
   - Real-time congestion predictions
   - Historical trend visualization
   - Confidence intervals

2. **Route Planner**
   - Multi-waypoint route optimization
   - Traffic-aware path selection
   - Dynamic ETA calculations
   - Alternative route suggestions

3. **Analytics Dashboard**
   - Weekly traffic heatmaps
   - Congestion distribution charts
   - Performance metrics monitoring
   - Anomaly detection alerts

4. **Monitoring System**
   - Real-time API health checks
   - Model performance tracking
   - System resource monitoring
   - Error logging and alerts

## 🛠️ Development

### Running Tests

```bash
# Frontend tests
cd traffic-cogestion-1
npm test

# Backend tests
cd traffic-cogestion-1/UCS_Model-main
python test_flask_api.py
python test_model_loading.py
```

### Code Quality

```bash
# Lint frontend
npm run lint

# Format code
npm run format

# Type checking
npm run type-check
```

### Building for Production

```bash
# Build frontend
cd traffic-cogestion-1
npm run build
npm start

# Production backend (with Gunicorn)
cd UCS_Model-main
gunicorn -w 4 -b 0.0.0.0:5000 traffic_prediction_api:app
```

## 🚢 Deployment

### Frontend (Vercel)

```bash
cd traffic-cogestion-1
npm install -g vercel
vercel deploy --prod
```

### Backend (Heroku/Railway)

```bash
cd traffic-cogestion-1/UCS_Model-main

# Create Procfile
echo "web: gunicorn traffic_prediction_api:app" > Procfile

# Deploy to Heroku
heroku create your-app-name
git push heroku main
```

### Docker Deployment

```bash
# Build and run with Docker Compose
docker-compose up -d

# Or build individually
docker build -t traffic-frontend ./traffic-cogestion-1
docker build -t traffic-backend ./traffic-cogestion-1/UCS_Model-main
```

## 📊 Performance Metrics

### Model Performance
- **Accuracy**: R² > 0.92
- **Error Rate**: RMSE ≤ 1.12
- **Response Time**: <100ms per prediction
- **Throughput**: 100+ requests/second

### System Performance
- **API Latency**: <200ms average
- **Frontend Load**: <2s initial load
- **Uptime Target**: 99.9%
- **Concurrent Users**: 100+ supported

## 🔧 Configuration

### Google Maps Setup

1. Enable required APIs in Google Cloud Console:
   - Maps JavaScript API
   - Directions API
   - Places API
   - Geocoding API
   - Traffic Layer

2. Add API key to `.env.local`
3. Configure billing and usage limits

### Supabase Setup (Optional)

1. Create Supabase project
2. Run database migrations
3. Configure Row Level Security
4. Add credentials to `.env.local`

## 🐛 Troubleshooting

### Common Issues

**Model Loading Errors**
```bash
# Verify model files exist
ls -la traffic-cogestion-1/UCS_Model-main/models/
# Should contain: traffic_prediction_model.h5, feature_scaler.pkl, target_scaler.pkl
```

**Port Conflicts**
```bash
# Kill process on port 5000
lsof -ti:5000 | xargs kill -9

# Or change port in traffic_prediction_api.py
```

**Google Maps Not Loading**
- Verify API key is correct and active
- Check API quotas and billing status
- Ensure all required APIs are enabled
- Check browser console for errors

**Python Dependencies**
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements_web.txt
```

## 📚 Documentation

- **Setup Guide**: `traffic-cogestion-1/README.md`
- **Model Documentation**: `traffic/README.md`
- **Running Instructions**: `traffic-cogestion-1/UCS_Model-main/README_Running_Instructions.txt`
- **Archive Notes**: `traffic/ARCHIVE_NOTE.md`

## 🧪 Testing

### Manual Testing

1. **Test API Health**
```bash
curl http://localhost:5000/api/health
```

2. **Test Prediction**
```bash
curl -X POST http://localhost:5000/api/ucs-predict \
  -H "Content-Type: application/json" \
  -d '{"latitude": 40.7128, "longitude": -74.0060, "timestamp": "2024-03-15T10:30:00Z"}'
```

3. **Test Frontend**
- Navigate to http://localhost:3000
- Enter location coordinates
- Verify prediction results
- Test route planning feature

## 🤝 Contributing

1. Fork the repository
2. Create feature branch: `git checkout -b feature/amazing-feature`
3. Commit changes: `git commit -m 'Add amazing feature'`
4. Push to branch: `git push origin feature/amazing-feature`
5. Open Pull Request

### Development Guidelines

- Follow TypeScript/Python best practices
- Write tests for new features
- Update documentation
- Ensure code passes linting
- Test locally before submitting PR

## 📄 License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **TensorFlow** - Deep learning framework
- **PyTorch Geometric** - Graph neural network library
- **Google Maps Platform** - Mapping and routing services
- **Next.js** - React framework
- **Supabase** - Database infrastructure
- **Vercel** - Hosting and deployment
- Open source community for various libraries
