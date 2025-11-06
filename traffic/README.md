# Traffic Prediction Model

This project implements deep learning models (LSTM with Attention and CNN-LSTM) for traffic prediction based on the Jupyter notebook `Cap_UCS_1.ipynb`.

## Features

- **LSTM with Attention**: Uses self-attention mechanism to focus on important time steps
- **CNN-LSTM Hybrid**: Combines convolutional layers with LSTM for feature extraction
- **Time Series Features**: Includes lag features, rolling statistics, and cyclical time encoding
- **Model Comparison**: Automatically compares models and saves the best performing one

## Requirements

- Python 3.7+
- TensorFlow 2.8+
- NumPy
- Pandas
- Matplotlib
- Scikit-learn

## Installation

1. Clone or download this repository
2. Install the required packages:

```bash
pip install -r requirements.txt
```

## Usage

1. **Prepare your dataset**: 
   - Ensure you have a CSV file named `final_dataset.csv` in the same directory
   - The dataset should contain traffic data with columns for timestamp, vehicle count, traffic speed, road occupancy, etc.

2. **Run the model**:

```bash
python traffic_prediction_model.py
```

## Dataset Format

The script expects a CSV file with the following columns:
- Timestamp
- Latitude
- Longitude  
- Vehicle_Count
- Traffic_Speed_kmh
- Road_Occupancy_Percent (target variable)
- Traffic_Light_State
- Weather_Condition
- Accident_Report
- Sentiment_Score
- Ride_Sharing_Demand
- Parking_Availability
- Emission_Levels_g_km
- Energy_Consumption_L_h
- Traffic_Condition

## Model Architecture

### LSTM with Attention
- Input layer for sequences
- LSTM layer (128 units) with return_sequences=True
- Self-attention mechanism
- LSTM layer (64 units)
- Dense layer (32 units, ReLU activation)
- Output layer (1 unit)

### CNN-LSTM Hybrid
- Conv1D layer (64 filters, kernel size 3)
- MaxPooling1D layer
- LSTM layer (100 units)
- Dropout layer (0.3)
- Dense layer (50 units, ReLU activation)
- Output layer (1 unit)

## Output

The script will:
1. Load and preprocess the data
2. Create time series sequences
3. Train both models
4. Display training progress and validation metrics
5. Show prediction plots for both models
6. Compare model performance (RMSE, MAE, R²)
7. Save the best model as `best_model.h5`

## Model Performance Metrics

- **RMSE**: Root Mean Square Error
- **MAE**: Mean Absolute Error  
- **R²**: Coefficient of Determination

## Customization

You can modify the following parameters in the `main()` function:

- `SEQUENCE_LENGTH`: Length of input sequences (default: 24)
- `PREDICTION_HORIZON`: How many steps ahead to predict (default: 1)
- `BATCH_SIZE`: Training batch size (default: 32)
- `EPOCHS`: Maximum training epochs (default: 100)

## Notes

- The script uses early stopping to prevent overfitting
- Learning rate reduction is applied when validation loss plateaus
- Random seeds are set for reproducibility
- The target variable (Road_Occupancy_Percent) is smoothed using a rolling window
- Cyclical encoding is used for time features (hour, day of week)

## Troubleshooting

1. **File not found error**: Ensure `final_dataset.csv` is in the same directory as the script
2. **Memory issues**: Reduce `BATCH_SIZE` or `SEQUENCE_LENGTH` if you encounter memory problems
3. **GPU usage**: TensorFlow will automatically use GPU if available and properly configured

## Original Source

This code was converted from the Jupyter notebook `Cap_UCS_1.ipynb` for easier local execution and deployment.