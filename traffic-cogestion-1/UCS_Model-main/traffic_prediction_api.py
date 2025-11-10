from flask import Flask, request, jsonify, render_template
import numpy as np
import pandas as pd
import tensorflow as tf
import joblib
import json
import os
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')
app = Flask(__name__)
model = None
feature_scaler = None
target_scaler = None
model_metadata = None
model_path = None
def load_model_and_scalers():
    global model, feature_scaler, target_scaler, model_metadata, model_path
    try:
        candidate_paths = [
            os.path.join('models', 'best_model.h5'),
            os.path.join('models', 'best_modlel.h5'),
            os.path.join('models', 'traffic_prediction_model.h5'),
            os.path.join('..', 'best_model.h5'),
            os.path.join('..', 'best_modlel.h5'),
            'best_model.h5',
            'best_modlel.h5',
        ]
        selected_path = None
        for path in candidate_paths:
            if os.path.exists(path):
                selected_path = path
                break
        if not selected_path:
            raise FileNotFoundError(
                "No model .h5 file found. Looked for: models/best_model.h5, models/traffic_prediction_model.h5, ../best_model.h5, best_model.h5"
            )
        model_path = selected_path
        print(f"📥 Loading model architecture from: {model_path} ...")
        model = tf.keras.models.load_model(model_path, compile=False)
        print("🔧 Compiling model with compatible metrics...")
        model.compile(
            optimizer='adam',
            loss='mse',
            metrics=['mae', 'mse']
        )
        print("📥 Loading feature and target scalers...")
        feature_scaler = joblib.load('models/feature_scaler.pkl')
        target_scaler = joblib.load('models/target_scaler.pkl')
        print("📥 Loading model metadata...")
        with open('models/model_metadata.json', 'r') as f:
            model_metadata = json.load(f)
        model_metadata['loaded_model_path'] = os.path.abspath(model_path) if model_path else None
        print("✅ Model and scalers loaded successfully!")
        print(f"   Model type: {model_metadata.get('model_type', 'Unknown')}")
        print(f"   Sequence length: {model_metadata.get('sequence_length', 'Unknown')}")
        print(f"   Features: {model_metadata.get('n_features', 'Unknown')}")
        try:
            input_shape = getattr(model, 'input_shape', None)
            if input_shape and len(input_shape) >= 3:
                seq_len_model = input_shape[1]
                n_features_model = input_shape[2]
                if 'sequence_length' in model_metadata and model_metadata['sequence_length'] != seq_len_model:
                    print(f"⚠️  sequence_length in metadata ({model_metadata['sequence_length']}) differs from model ({seq_len_model}). Using model value.")
                    model_metadata['sequence_length'] = int(seq_len_model)
                if 'n_features' in model_metadata and model_metadata['n_features'] != n_features_model:
                    print(f"⚠️  n_features in metadata ({model_metadata['n_features']}) differs from model ({n_features_model}). Using model value.")
                    model_metadata['n_features'] = int(n_features_model)
            if hasattr(feature_scaler, 'n_features_in_'):
                print(f"   Feature scaler expects: {feature_scaler.n_features_in_} features")
        except Exception as _:
            pass
        return True
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        print(f"   Make sure you're in the correct directory (UCS_Model-main)")
        print(f"   and that all model files exist in the 'models/' folder")
        return False
def preprocess_location_data(lat, lon, timestamp, additional_features=None):
    dt = pd.to_datetime(timestamp)
    hour = dt.hour
    dow = dt.dayofweek
    is_weekend = 1 if dow in [5, 6] else 0
    hour_sin = np.sin(2 * np.pi * hour / 24)
    hour_cos = np.cos(2 * np.pi * hour / 24)
    lat_norm = (lat - 16.5) / 2.0
    lon_norm = (lon - 80.5) / 2.0
    features = np.array([
        lat_norm, lon_norm,
        0, 0,
        0, 0, 0,
        0, 0, 0,
        hour, dow, is_weekend,
        hour_sin, hour_cos,
        0, 0,
        0, 0, 0,
        0, 0
    ])
    return features
def predict_traffic_for_location(lat, lon, timestamp, hours_ahead=1):
    try:
        features = preprocess_location_data(lat, lon, timestamp)
        sequence_length = model_metadata['sequence_length']
        sequence = np.tile(features, (sequence_length, 1))
        sequence_scaled = feature_scaler.transform(sequence)
        sequence_scaled = sequence_scaled[:, :18]
        sequence_scaled = sequence_scaled.reshape(1, sequence_length, -1)
        pred_scaled = model.predict(sequence_scaled, verbose=0)
        pred_original = target_scaler.inverse_transform(pred_scaled.reshape(-1, 1))[0][0]
        pred_original = max(0, min(100, pred_original))
        dt = pd.to_datetime(timestamp)
        hour = dt.hour
        minute = dt.minute
        if (hour >= 7 and hour <= 9) or (hour >= 17 and hour <= 19):
            base_multiplier = 1.3
            minute_factor = 1.0 + (30 - abs(30 - minute)) / 150
            pred_original = min(100, pred_original * base_multiplier * minute_factor + 10)
        elif (hour >= 10 and hour <= 16):
            minute_factor = 1.0 + (minute / 300)
            pred_original = min(100, pred_original * 1.15 * minute_factor + 5)
        elif hour >= 0 and hour <= 5:
            pred_original = max(0, pred_original * 0.3)
        else:
            minute_factor = 1.0 + (minute / 600)
            pred_original = min(100, pred_original * 1.05 * minute_factor)
        distance_from_center = np.sqrt((lat - 16.5)**2 + (lon - 80.6)**2)
        if distance_from_center < 0.05:
            pred_original = min(100, pred_original * 1.15 + 8)
        elif distance_from_center > 0.2:
            pred_original = max(0, pred_original * 0.7 - 5)
        variation_seed = int((lat * 1000 + lon * 1000) % 100)
        if variation_seed % 3 == 0:
            pred_original = max(0, pred_original * 0.85)
        elif variation_seed % 3 == 1:
            pred_original = min(100, pred_original * 1.1)
        return {
            'prediction': float(pred_original),
            'confidence': 'high' if pred_original > 50 else 'medium',
            'timestamp': timestamp,
            'location': {'lat': lat, 'lon': lon},
            'factors': {
                'hour': hour,
                'is_peak_hour': (hour >= 7 and hour <= 9) or (hour >= 17 and hour <= 19),
                'distance_from_center_km': float(distance_from_center * 111)
            }
        }
    except Exception as e:
        return {'error': str(e)}
@app.route('/')
def index():
    return render_template('index.html')
@app.route('/api/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        required_fields = ['latitude', 'longitude', 'timestamp']
        for field in required_fields:
            if field not in data:
                return jsonify({'error': f'Missing required field: {field}'}), 400
        lat = float(data['latitude'])
        lon = float(data['longitude'])
        timestamp = data['timestamp']
        if not (-90 <= lat <= 90) or not (-180 <= lon <= 180):
            return jsonify({'error': 'Invalid coordinates'}), 400
        result = predict_traffic_for_location(lat, lon, timestamp)
        if 'error' in result:
            return jsonify({'error': result['error']}), 500
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500
@app.route('/api/predict_route', methods=['POST'])
def predict_route():
    try:
        data = request.get_json()
        if 'waypoints' not in data:
            return jsonify({'error': 'Missing waypoints'}), 400
        waypoints = data['waypoints']
        if len(waypoints) < 2:
            return jsonify({'error': 'At least 2 waypoints required'}), 400
        route_predictions = []
        for i, waypoint in enumerate(waypoints):
            if 'latitude' not in waypoint or 'longitude' not in waypoint:
                return jsonify({'error': f'Invalid waypoint {i}'}), 400
            base_time = datetime.now()
            waypoint_time = base_time + timedelta(minutes=i * 5)
            result = predict_traffic_for_location(
                waypoint['latitude'], 
                waypoint['longitude'], 
                waypoint_time.isoformat()
            )
            if 'error' not in result:
                route_predictions.append({
                    'waypoint': i,
                    'location': waypoint,
                    'prediction': result['prediction'],
                    'timestamp': waypoint_time.isoformat()
                })
        if route_predictions:
            avg_traffic = np.mean([p['prediction'] for p in route_predictions])
            max_traffic = max([p['prediction'] for p in route_predictions])
            min_traffic = min([p['prediction'] for p in route_predictions])
            return jsonify({
                'route_predictions': route_predictions,
                'summary': {
                    'average_traffic': float(avg_traffic),
                    'max_traffic': float(max_traffic),
                    'min_traffic': float(min_traffic),
                    'total_waypoints': len(route_predictions)
                }
            })
        else:
            return jsonify({'error': 'No valid predictions'}), 500
    except Exception as e:
        return jsonify({'error': str(e)}), 500
@app.route('/api/model_info', methods=['GET'])
def model_info():
    if model_metadata is None:
        return jsonify({'error': 'Model not loaded'}), 500
    return jsonify(model_metadata)
@app.route('/api/health', methods=['GET'])
def health():
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'timestamp': datetime.now().isoformat()
    })
if __name__ == '__main__':
    if load_model_and_scalers():
        print("🚀 Starting Traffic Prediction API...")
        print("📊 Model loaded and ready for predictions!")
        print("🌐 API will be available at: http://localhost:5001")
        print("📱 Web interface at: http://localhost:5001")
        app.run(debug=True, host='0.0.0.0', port=5001)
    else:
        print("❌ Failed to load model. Please check model files.")