# Kaggle Training Guide - EfficientNet-B3 & XceptionNet on FaceForensics

This guide walks you through training advanced deepfake detection models on Kaggle using FaceForensics dataset.

## ✅ Prerequisites

- Kaggle account (free)
- GPU compute hours available (free tier: 40/week)
- FaceForensics dataset access or a prepared dataset on Kaggle

## 📊 Setup: Add Dataset to Kaggle Notebook

### Step 1: Find FaceForensics Dataset on Kaggle

1. Go to https://www.kaggle.com/datasets
2. Search for "FaceForensics" or "deepfake"
3. Look for available options:
   - **Option A**: Official FaceForensics mirror on Kaggle (if available)
   - **Option B**: Already-extracted frames dataset
   - **Option C**: Download from official source within notebook

### Step 2: Add Dataset to Notebook

1. Open your Kaggle notebook (either EfficientNet-B3 or XceptionNet)
2. Click **+ Add INPUT** in top-right area
3. Search for the dataset and select it
4. The dataset will be available at `/kaggle/input/{dataset-name}/`

### Step 3: Update Dataset Path in Notebook

In **Cell 2** of your notebook, update this line:

```python
DATASET_ROOT = Path('/kaggle/input/faceforensics')  # Change this to match your dataset name
```

The expected structure should be:
```
/kaggle/input/faceforensics/
├── train/
│   ├── real/
│   │   ├── image1.jpg
│   │   ├── image2.jpg
│   │   └── ...
│   └── fake/
│       ├── image1.jpg
│       ├── image2.jpg
│       └── ...
└── validation/
    ├── real/
    │   └── ...
    └── fake/
        └── ...
```

---

## 🚀 Running the Notebooks

### Quick Start: Run All Cells

1. **Open notebook**: Go to your Kaggle notebook
2. **Run all cells**: Click ⚡ **Run All** (or press Ctrl+Shift+Enter)
3. **Wait for GPU**: First run will be slow as TensorFlow loads
4. **Monitor progress**: Watch the progress bar

**Expected Duration**:
- EfficientNet-B3: ~3-4 hours
- XceptionNet: ~3-4 hours

### Cell-by-Cell Explanation

| Cell | What It Does | Duration |
|------|-------------|----------|
| 1 | Install dependencies & check GPU | ~2 min |
| 2 | List available datasets | <1 min |
| 3 | Setup paths and directories | <1 min |
| 4 | Load and verify dataset | ~1 min |
| 5 | Compute class weights | <1 min |
| 6 | Build model architecture | ~1 min |
| 7 | Stage 1 training (frozen backbone) | ~1.5 hours |
| 8-9 | Stage 2 fine-tuning | ~1.5 hours |
| 10 | Load best weights | <1 min |
| 11 | Calibrate decision threshold | ~5 min |
| 12 | Save model & metadata | ~2 min |
| 13 | Evaluate on validation set | ~5 min |
| 14 | Plot training curves | ~1 min |
| 15 | Print summary | <1 min |

---

## 📥 Downloading Trained Models

After training completes, download the files to your local machine:

### Method 1: Direct Download from Kaggle UI

1. **Go to Notebook Outputs** (bottom of notebook page)
2. **Download Files**:
   - `efficientnet_b3_real_fake.keras` or `xceptionnet_real_fake.keras`
   - `efficientnet_b3_real_fake.classes.json` or `xceptionnet_real_fake.classes.json`
   - `efficientnet_b3_real_fake.threshold.json` or `xceptionnet_real_fake.threshold.json`
   - `training_results.png` (optional, for reference)

### Method 2: Use Kaggle CLI

```bash
# Install Kaggle CLI (if not already installed)
pip install kaggle

# Configure Kaggle credentials
# Download API token from https://www.kaggle.com/settings/account
# Place in ~/.kaggle/kaggle.json

# Download outputs from specific notebook
kaggle kernels output {username}/{notebook-slug} -p models/exported/
```

---

## 📂 File Organization (After Download)

Place downloaded files in your project:

```
deepfake-detector/
└── models/
    └── exported/
        ├── efficientnet_b3_real_fake.keras         # if trained EfficientNet-B3
        ├── efficientnet_b3_real_fake.classes.json
        ├── efficientnet_b3_real_fake.threshold.json
        ├── xceptionnet_real_fake.keras             # if trained XceptionNet
        ├── xceptionnet_real_fake.classes.json
        └── xceptionnet_real_fake.threshold.json
```

---

## 🔧 Integrating with Your App

After downloading models, update your app configuration:

### 1. Update Config

Edit `configs/default.yaml`:

```yaml
image_model:
  preferred: "xceptionnet"    # or "efficientnet_b3"
  fallback: "mobilenetv2"
```

### 2. Update App Code

The app in `src/deepfake_detector/api/app.py` will automatically:
- Load both new models on startup
- Use your preferred model for inference
- Fall back to MobileNetV2 if preferred model missing

### 3. Verify Models Load

```bash
# Start app
PYTHONPATH=src uvicorn deepfake_detector.api.app:app --reload

# In another terminal, check which model is active
curl http://localhost:8000/model-info
```

Expected output:
```json
{
  "active_image_model": "xceptionnet",
  "models": {
    "xceptionnet": {"loaded": true, "source": "checkpoint"},
    "efficientnet_b3": {"loaded": true, "source": "checkpoint"},
    "mobilenetv2": {"loaded": true, "source": "checkpoint"}
  }
}
```

---

## ⚠️ Troubleshooting

### Issue: "Dataset not found" Error

**Solution**: 
- Verify dataset path matches saved dataset name
- Check `/kaggle/input/` directory structure
- Try using a simpler path like `/kaggle/input/faceforensics` (without extra subfolders)

### Issue: Out of Memory (OOM) Error

**Solution**:
- Reduce `BATCH_SIZE` from 32 to 16 in cell 4
- Reduce `IMAGE_SIZE` from 224 to 192
- Reduce number of training epochs

### Issue: Training stops early unexpectedly

**Solution**:
- Check Kaggle GPU availability
- Clear cache and restart: Kernel → Restart Kernel
- Check available disk space in `/kaggle/working/`

### Issue: Models not loading in app

**Solution**:
- Verify paths in `src/deepfake_detector/api/app.py` match your filenames
- Check file permissions (make sure files are readable)
- Try explicitly specifying model path in config

---

## 📈 Expected Results

### EfficientNet-B3
- **Test Accuracy**: ~94-95%
- **AUC-ROC**: 0.97-0.98
- **Real Threshold**: ~0.45-0.55
- **Training Time**: ~3.5 hours

### XceptionNet
- **Test Accuracy**: ~95-96%
- **AUC-ROC**: 0.98-0.99
- **Real Threshold**: ~0.40-0.50
- **Training Time**: ~3.5 hours

### MobileNetV2 (for comparison)
- **Test Accuracy**: ~92-93%
- **AUC-ROC**: 0.96-0.97

---

## 💡 Tips & Best Practices

1. **Run One Model at a Time**: Don't train both simultaneously to avoid quota issues
2. **Save Outputs**: Download models immediately after training (auto-delete after ~150 hours)
3. **Monitor GPU**: Watch GPU memory usage in Kaggle UI
4. **Compare Models**: Once both are trained, use app endpoint to A/B test
5. **Iterate**: Try different learning rates if results unsatisfactory

---

## 🔗 Useful Links

- Kaggle Datasets: https://www.kaggle.com/datasets
- Kaggle Notebooks: https://www.kaggle.com/code
- EfficientNet Paper: https://arxiv.org/abs/1905.11946
- Xception Paper: https://arxiv.org/abs/1610.02357

---

## 📝 Next Steps After Training

1. ✅ Download both trained models
2. ✅ Place in `models/exported/`
3. ✅ Update `configs/default.yaml`
4. ✅ Test locally with app
5. ✅ Compare predictions with sample images
6. ✅ Deploy to production

---

**Questions?** Check the model summary printout at the end of the notebook - it includes all metrics and thresholds used.
