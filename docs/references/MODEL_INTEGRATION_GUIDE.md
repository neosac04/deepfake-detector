# Model Integration Guide

## Overview
This guide outlines the integration of trained MobileNetV2 and EfficientNet models into the deepfake detector application. Both models are CNN-based image classifiers trained to distinguish between **Real** and **Fake** media.

## Trained Models Status

### MobileNetV2
- **File**: `mobilenetv2_deepfake.keras`
- **Format**: TensorFlow SavedModel
- **Classes**: Fake (0), Real (1)
- **Optimal Threshold**: 0.39
- **Performance Metrics**:
  - Accuracy: 81.53%
  - AUC-ROC: 90.60%
  - Test Samples: 10,905
- **Source**: Desktop/Results/MobileNetv2/

### EfficientNet (B3)
- **File**: `efficientnet_real_fake.keras`
- **Format**: TensorFlow SavedModel
- **Classes**: Fake (0), Real (1)
- **Optimal Threshold**: 0.46
- **Source**: Desktop/Results/

## Integration Architecture

### Current App Infrastructure
The FastAPI backend (`backend/app.py`) already supports:
- TensorFlow model loading via `tf_load_model()`
- Image classification through `_predict_image_tf()`
- Metadata loading from JSON configuration files
- Threshold-based decision making

### Model File Organization
```
models/
├── efficientnet/
│   ├── efficientnet_real_fake.keras
│   ├── efficientnet_real_fake.classes.json
│   ├── efficientnet_real_fake.threshold.json
│   └── efficientnet_real_fake.metrics.json
└── mobilenetv2/
    ├── mobilenetv2_deepfake.keras
    ├── mobilenetv2_deepfake.classes.json
    ├── mobilenetv2_deepfake.threshold.json
    └── mobilenetv2_deepfake.metrics.json
```

### Metadata File Specifications

#### Classes File (e.g., `{model_name}.classes.json`)
```json
{
  "Fake": 0,
  "Real": 1
}
```

#### Threshold File (e.g., `{model_name}.threshold.json`)
```json
{
  "threshold": 0.39
}
```

#### Metrics File (e.g., `{model_name}.metrics.json`)
```json
{
  "accuracy": 0.8153,
  "auc_roc": 0.9060,
  "test_samples": 10905,
  "threshold": 0.39
}
```

## Implementation Plan

### Phase 1: Setup Model Files ✅
1. Copy trained models from Desktop/Results to models/ directory
2. Organize by model type (MobileNetV2, EfficientNet)
3. Create standardized metadata files for each model
4. Ensure all metadata uses correct class names: **Real** and **Fake**

### Phase 2: Update Backend Configuration
1. Modify `backend/app.py` to support multi-model loading
2. Create model registry to track available models
3. Implement model selection via query parameters
4. Update startup event to load all available models

**Key Changes:**
- Add model registry dictionary to store loaded models and their metadata
- Create helper function `load_available_models()` to discover and load all models
- Update prediction logic to select model based on parameter (default: MobileNetV2)

### Phase 3: Extend API Endpoints
Create new endpoints for model management:

#### `GET /models/available`
Returns list of available models with metadata
```json
{
  "models": [
    {
      "name": "mobilenetv2",
      "accuracy": 0.8153,
      "auc_roc": 0.9060,
      "threshold": 0.39,
      "test_samples": 10905
    },
    {
      "name": "efficientnet",
      "accuracy": "...",
      "auc_roc": "...",
      "threshold": 0.46,
      "test_samples": "..."
    }
  ]
}
```

#### `POST /predict?model=mobilenetv2`
Predict with specified model (defaults to mobilenetv2)
```json
{
  "prediction": "Real",
  "confidence": 0.87,
  "threshold": 0.39,
  "model": "mobilenetv2"
}
```

### Phase 4: Ensure Balanced Output
To guarantee both models produce balanced, reliable classifications:

1. **Threshold Application**:
   - MobileNetV2: Use 0.39 threshold
   - EfficientNet: Use 0.46 threshold
   - Output "Real" if confidence ≥ threshold, else "Fake"

2. **Confidence Calculation**:
   - Model outputs probability (0-1) for Real class
   - Confidence = max(prob_real, 1 - prob_real)
   - Ensures confidence is always [0, 1] and represents prediction strength

3. **Model Comparison Mode** (optional):
   - Run both models on same input
   - Return ensemble prediction: "Real" if both agree, "Uncertain" if conflicting
   - Improves overall reliability

4. **Logging and Monitoring**:
   - Log each prediction with model name, raw scores, and applied threshold
   - Track prediction distribution to detect model drift
   - Alert if one model consistently differs from the other

### Phase 5: Frontend Updates
1. Add model selector dropdown (default: MobileNetV2)
2. Display selected model's performance metrics before prediction
3. Show applied threshold value in results
4. Add confidence indicator visualization
5. Optional: Show both model predictions for comparison

## Class Labels Reference
**Important**: The dataset uses capitalized class names:
- **Real**: Label 1 - Genuine/authentic media
- **Fake**: Label 0 - Deepfake/synthesized media

All JSON metadata files and predictions must use these exact capitalizations.

## Validation Checklist

- [ ] Models copied to `models/` directory structure
- [ ] All metadata JSON files created with correct class capitalizations
- [ ] Backend loads both models successfully on startup
- [ ] Model selection endpoint works with query parameters
- [ ] Both models produce predictions on test images
- [ ] Predictions respect optimal thresholds
- [ ] Confidence scores are in [0, 1] range
- [ ] Frontend displays model selection correctly
- [ ] Performance metrics displayed accurately
- [ ] Ensemble mode (if enabled) works correctly

## Testing Procedure

1. **Individual Model Tests**:
   ```bash
   # Test MobileNetV2
   curl -X POST "http://localhost:8000/predict?model=mobilenetv2" \
     -F "file=@test_image.jpg"
   
   # Test EfficientNet
   curl -X POST "http://localhost:8000/predict?model=efficientnet" \
     -F "file=@test_image.jpg"
   ```

2. **Model Comparison**:
   - Test same image with both models
   - Verify predictions align within confidence tolerance
   - Log any significant divergences

3. **Threshold Validation**:
   - Test edge cases near thresholds (0.38-0.41 for MobileNetV2, 0.45-0.47 for EfficientNet)
   - Verify predictions flip at exact threshold values
   - Ensure confidence calculations are correct

4. **Balance Check**:
   - Run on diverse test set (Real and Fake examples)
   - Verify both models classify correctly
   - Check for bias toward one class

## Performance Expectations

| Metric | MobileNetV2 | EfficientNet |
|--------|-------------|--------------|
| Accuracy | 81.53% | TBD |
| AUC-ROC | 90.60% | TBD |
| Threshold | 0.39 | 0.46 |
| Test Samples | 10,905 | TBD |

## Troubleshooting

### Model Loading Issues
- Ensure `.keras` files are valid TensorFlow SavedModels
- Check file paths match configuration exactly
- Verify JSON metadata files are valid JSON

### Prediction Mismatches
- Confirm thresholds match those used during model validation
- Check input image preprocessing matches training pipeline
- Verify class indices are correct (Fake=0, Real=1)

### Confidence Anomalies
- If confidence < 0.5, model is uncertain
- Check if input is similar to training distribution
- Consider ensemble prediction if models disagree

## References
- Training Guide: [KAGGLE_TRAINING.md](KAGGLE_TRAINING.md)
- Model Architecture: See `backend/models/` for model definitions
- Data Preparation: See `backend/utilities/data/` for preprocessing logic
