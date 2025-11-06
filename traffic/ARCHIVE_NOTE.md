# Traffic Models - Archive

## ⚠️ NOTICE: This folder is now archived

The models and scripts from this folder have been **successfully integrated** into the main application at:
```
traffic-cogestion-1/UCS_Model-main/
```

## What's in this folder?

This folder contains the **original training code and models**:

### Model Files:
- `lstm_model.h5` - Trained LSTM model
- `cnn_lstm_model.h5` - Trained CNN-LSTM model

### Training Scripts:
- `lstm_model.py` - LSTM model training
- `cnn_lstm_model.py` - CNN-LSTM model training
- `combine_models.py` - Model comparison and ensemble

### Data & Documentation:
- `final_dataset.csv` - Training dataset
- `Cap_UCS_1.ipynb` - Original Jupyter notebook
- `requirements.txt` - Python dependencies
- `README.md` - Original documentation

## Where are the models now?

✅ **Integrated Location**: `traffic-cogestion-1/UCS_Model-main/`

The models have been copied to:
```
traffic-cogestion-1/UCS_Model-main/models/
├── lstm_model.h5
├── cnn_lstm_model.h5
└── best_model.h5
```

Training scripts are available at:
```
traffic-cogestion-1/UCS_Model-main/
├── lstm_model.py
├── cnn_lstm_model.py
└── combine_models.py
```

## How to use the integrated system?

See the main application documentation:
- **Integration Guide**: `traffic-cogestion-1/INTEGRATION_SUMMARY.md`
- **Usage Guide**: `traffic-cogestion-1/UCS_Model-main/README_INTEGRATED.md`
- **Quick Start**: `traffic-cogestion-1/run.txt`

## Can I delete this folder?

**Recommendation**: Keep this folder as a backup/archive of the original training code.

If you need to:
- ✅ **Keep**: For reference, backup, or retraining from scratch
- ❌ **Delete**: If you're confident the integration is complete and working

## Retraining Models

If you want to retrain models, you can either:

1. **Use this folder** (original training environment):
   ```bash
   cd traffic
   python lstm_model.py
   python cnn_lstm_model.py
   ```

2. **Use integrated scripts** (recommended):
   ```bash
   cd traffic-cogestion-1/UCS_Model-main
   python lstm_model.py
   python cnn_lstm_model.py
   ```

---

**Archive Date**: November 2024  
**Status**: ✅ Successfully Integrated  
**Safe to Delete**: Yes (but recommended to keep as backup)
