# 🚗 Traffic Congestion Prediction System

A comprehensive AI-powered traffic prediction platform that combines advanced machine learning models (UCS Hybrid Architecture) with real-time visualization and intelligent route planning capabilities.

## 🌟 Features

### 🤖 Advanced AI Models (UCS)
- **Hybrid Architecture**: Combines LSTM (Long Short-Term Memory), CNN (Convolutional Neural Networks), and GRU (Gated Recurrent Units) for robust temporal and spatial modeling.
- **Spatio-temporal Analysis**: Graph Neural Networks (GNN) for understanding complex road network dependencies.
- **Real-time Processing**: Sub-200ms prediction response latency.
- **High Accuracy**: ~92% prediction accuracy on test datasets.
- **Input Features**: Utilizes 32 distinct features including historical flow, weather conditions, and temporal indicators.

### 🗺️ Interactive Dashboard & Route Planner
- **Live Traffic Visualization**: Real-time traffic overlay on Google Maps with color-coded congestion levels.
- **Smart Route Planning**: 
  - Multi-waypoint routing.
  - Predictive ETA calculations.
  - Congestion-aware pathfinding.
- **Data Analytics**:
  - Weekly traffic heatmaps.
  - Congestion distribution analysis.
  - Historical traffic trends chart.
  - Model comparison metrics.
- **Backend Health Monitoring**: Real-time status checks for the ML API.

### 🔐 Secure & User-Friendly
- **Authentication**: Secure Sign-in/Sign-up flow with AuthGuard protection.
- **Modern UI**: Dark-themed, responsive interface built with Next.js 16 and Tailwind CSS 4.
- **Project Info Card**: Quick access to developer details and project context.

## 🔧 Technical Stack

### Frontend (`/view`)
- **Framework**: [Next.js 16](https://nextjs.org/) (App Router, Turbopack)
- **Language**: TypeScript
- **Styling**: Tailwind CSS 4, Radix UI, Lucide React
- **Maps**: Google Maps JavaScript API, @react-google-maps/api
- **Charts**: Recharts

### Backend (`/UCS_Model-main`)
- **API**: Flask (Python)
- **ML Frameworks**: TensorFlow/Keras, PyTorch Geometric, scikit-learn
- **Data Processing**: Pandas, NumPy

### Database & Infrastructure
- **Database**: Supabase (PostgreSQL)
- **Deployment**: Vercel (Frontend), Heroku/Render (Backend recommended)

## 🚀 Quick Start

### Prerequisites
- Node.js 18+ and npm/pnpm
- Python 3.9+ with pip
- Google Maps API key (with Maps JS, Places, Directions, Geocoding APIs enabled)
- Supabase account

### 1. Clone and Setup

```bash
git clone <repository-url>
cd traffic-cogestion-1
```

### 2. Backend Setup
```bash
cd UCS_Model-main
pip install -r requirements_web.txt

# Start the Flask API
python traffic_prediction_api.py
```
_The backend runs on http://localhost:5000_

### 3. Frontend Setup
```bash
# Open a new terminal
cd view

# Install dependencies
npm install

# Create .env.local
cp .env.example .env.local
# (Edit .env.local with your Google Maps API Key and Supabase Config)

# Start the development server
npm run dev
```
_The frontend runs on http://localhost:3000_

## 📊 Application Structure

```
traffic-cogestion-1/
├── view/                        # Next.js Frontend Application
│   ├── src/
│   │   ├── app/                 # App Router (Pages & Layouts)
│   │   │   ├── auth/            # Authentication Routes
│   │   │   ├── dashboard/       # Main Traffic Dashboard
│   │   │   ├── route-planner/   # Route Optimization Tool
│   │   │   └── ...
│   │   ├── components/          # Reusable UI Components
│   │   │   ├── ui/              # Radix UI Primitives
│   │   │   ├── traffic-map.tsx  # Map Visualization
│   │   │   ├── project-info.tsx # Developer Info Card
│   │   │   └── ...
│   │   └── lib/                 # Utilities & Helpers
│   └── ...
├── UCS_Model-main/              # Python/Flask ML Backend
│   ├── models/                  # Saved .h5 and .pkl files
│   ├── traffic_prediction_api.py # Main API Entry Point
│   └── ...
└── README.md                    # Project Documentation
```

## 🔌 Core API Endpoints

- `POST /api/ucs-predict`: Get traffic predictions for a specific location.
- `POST /api/ucs-predict-route`: Analyze and predict traffic for a route.
- `GET /api/ucs-model-info`: Retrieve current model status and metrics.
- `GET /api/health`: System health check.

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/NewFeature`)
3. Commit your changes (`git commit -m 'Add NewFeature'`)
4. Push to the branch (`git push origin feature/NewFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the Apache License 2.0.

## 📞 Contact

**Sairam Maruri**  
- 📧 sairam.maruri@gmail.com
- 🌐 [Portfolio](https://saiii.in)
- 🐙 [GitHub](https://github.com/sairam3824)
