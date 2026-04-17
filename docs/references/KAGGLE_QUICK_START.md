# Kaggle Training - Quick Reference

## 📋 What Has Been Created

### 1. Kaggle Notebooks (Ready to Use)
✅ `notebooks/kaggle_efficientnet_b3_faceforensics.ipynb`
- Complete training pipeline for EfficientNet-B3
- 15 cells covering data loading → model saving → evaluation
- Copy-paste ready into Kaggle

✅ `notebooks/kaggle_xceptionnet_faceforensics.ipynb`
- Complete training pipeline for XceptionNet
- Identical structure to EfficientNet notebook
- Copy-paste ready into Kaggle

### 2. Documentation
✅ `docs/KAGGLE_TRAINING.md`
- Step-by-step guide for running on Kaggle
- Dataset setup instructions
- Troubleshooting section
- Expected results and timelines

---

## 🎯 Quick Workflow (5 Steps)

### Step 1: Copy Notebook to Kaggle
1. Go to https://www.kaggle.com/code
2. Create new notebook
3. Copy content from `notebooks/kaggle_efficientnet_b3_faceforensics.ipynb`
4. Paste into Kaggle notebook cells
5. Save notebook

### Step 2: Add FaceForensics Dataset
1. Click **+ Add INPUT**
2. Search for "faceforensics" or "deepfake"
3. Select dataset and add
4. Update path in notebook Cell 3 if needed

### Step 3: Run Training
1. Click **⚡ Run All**
2. Wait 3-4 hours for GPU training
3. Watch progress in notebook output
4. Models saved to `/kaggle/working/output/`

### Step 4: Download Models
1. Bottom of notebook → Outputs section
2. Download 3 files:
   - `efficientnet_b3_real_fake.keras`
   - `efficientnet_b3_real_fake.classes.json`
   - `efficientnet_b3_real_fake.threshold.json`

### Step 5: Integrate with App
1. Move downloaded files to `models/exported/`
2. Update `configs/default.yaml`:
   ```yaml
   image_model:
     preferred: "efficientnet_b3"
   ```
3. Start app: `PYTHONPATH=src uvicorn deepfake_detector.api.app:app --reload`
4. Open [http://localhost:8000](http://localhost:8000)

---

## 📊 What Each Notebook Does

### Both Notebooks Include:

| Cell | Purpose |
|------|---------|
| 1 | Install TensorFlow, check GPU |
| 2 | Verify Kaggle datasets available |
| 3 | Setup paths and output directory |
| 4 | Create data generators with augmentation |
| 5 | Calculate class weights for imbalance |
| 6 | Build model (EfficientNet-B3 or Xception) |
| 7 | **Stage 1**: Train head (frozen backbone) |
| 8-9 | **Stage 2**: Fine-tune top 30 backbone layers |
| 10 | Load best weights from training |
| 11 | Calibrate optimal decision threshold |
| 12 | Save `.keras` model + metadata |
| 13 | Classification report, AUC, confusion matrix |
| 14 | Plot training curves |
| 15 | Print final summary |

---

## 🚀 Expected Metrics

| Model | Accuracy | AUC | Time | Size |
|-------|----------|-----|------|------|
| **EfficientNet-B3** | ~95% | 0.97 | 3.5h | ~47MB |
| **XceptionNet** | ~96% | 0.98 | 3.5h | ~83MB |
| MobileNetV2 (current) | ~92% | 0.96 | - | ~36MB |

---

## 📦 Files Downloaded Per Model

After training completes and downloads, you'll have:

```
models/exported/
├── efficientnet_b3_real_fake.keras          (~47 MB)
├── efficientnet_b3_real_fake.classes.json   (~50 B)
└── efficientnet_b3_real_fake.threshold.json (~30 B)

# AND/OR

├── xceptionnet_real_fake.keras          (~83 MB)
├── xceptionnet_real_fake.classes.json   (~50 B)
└── xceptionnet_real_fake.threshold.json (~30 B)
```

---

## ✨ App Integration (Auto-Handled)

The app code (`src/deepfake_detector/api/app.py`) automatically:

1. **Detects** if `.keras` files exist at startup
2. **Loads** all available models (XceptionNet, EfficientNet-B3, MobileNetV2)
3. **Uses** preferred model from config (or best available)
4. **Falls back** gracefully if a model missing
5. **Provides** `/model-info` endpoint to see all loaded models

**No code changes needed!** Just place files and update config.

---

## 🔍 Verification After Download

```bash
# 1. Check files are there
ls -lh models/exported/*.keras

# 2. Start app
PYTHONPATH=src uvicorn deepfake_detector.api.app:app --reload

# 3. Check which model is active
curl http://localhost:8000/model-info

# 4. Test on sample image
curl -X POST -F "file=@test_image.jpg" http://localhost:8000/predict
```

---

## ⚡ Cost & Time Summary

| Resource | Cost | Time |
|----------|------|------|
| GPU hours | Free (40/week) | 3.5h per model |
| Storage | Free | ~130 MB total |
| Kaggle account | Free | - |
| FaceForensics access | Free (academic) | - |

**Total for both models**: ~7 hours GPU, <200MB storage

---

## 🎓 Training Overview

### Stage 1 (Hours 0-1.5): Train Head Only
```
Backbone: FROZEN (ImageNet weights)
Head: TRAINING
Learning Rate: 1e-4
Epochs: 10
```

### Stage 2 (Hours 1.5-3): Fine-tune Backbone
```
Backbone: TOP 30 LAYERS UNFROZEN
Head: TRAINING
Learning Rate: 1e-5
Epochs: 10
```

### Result
```
✓ Better accuracy than frozen baseline
✓ Optimized for FaceForensics patterns
✓ Ready for deployment
```

---

## 📖 Next: Read Full Guide

For detailed setup instructions, troubleshooting, and dataset info:
👉 See `docs/KAGGLE_TRAINING.md`

---

## ❓ Common Questions

**Q: Do I need to download the dataset locally?**
No! Training happens entirely on Kaggle. Only download the trained `.keras` files.

**Q: Can I run both notebooks simultaneously?**
Not recommended - you'll hit GPU quota. Train one, then the other.

**Q: Will MobileNetV2 still work?**
Yes! App keeps all 3 models and falls back automatically.

**Q: How often should I retrain?**
New FaceForensics samples released quarterly. Retrain when needed.

**Q: Can I use custom dataset?**
Yes! Just upload to Kaggle and mount in notebook instead of FaceForensics.

---

## 🎯 You Are Here

```
Create notebooks ✓ ← YOU ARE HERE
    ↓
Add dataset to Kaggle
    ↓
Run training (3-4h per model)
    ↓
Download .keras files
    ↓
Place in models/exported/
    ↓
Update configs/default.yaml
    ↓
Test app locally
    ↓
Deploy! 🚀
```

---

**Ready to start? Copy one of the `.ipynb` files to Kaggle and follow `docs/KAGGLE_TRAINING.md`**
