# Advanced Training Guide: EfficientNet & XceptionNet on FaceForensics

This guide walks you through training high-performance deepfake detection models using modern architectures (EfficientNet, XceptionNet) on the FaceForensics dataset. Each section contains 3 actionable steps organized by complexity.

---

## Table of Contents
1. [Part 1: Project Foundation](#part-1-project-foundation)
2. [Part 2: Dataset Setup](#part-2-dataset-setup)
3. [Part 3: Understanding Advanced Models](#part-3-understanding-advanced-models)
4. [Part 4: Data Preparation](#part-4-data-preparation)
5. [Part 5: Building Training Infrastructure](#part-5-building-training-infrastructure)
6. [Part 6: Advanced Augmentation](#part-6-advanced-augmentation)
7. [Part 7: Transfer Learning & Fine-tuning](#part-7-transfer-learning--fine-tuning)
8. [Part 8: Training & Monitoring](#part-8-training--monitoring)
9. [Part 9: Evaluation & Metrics](#part-9-evaluation--metrics)
10. [Part 10: Production Optimization](#part-10-production-optimization)

---

## Part 1: Project Foundation

**What you'll learn:** Understand the current project structure and existing implementations before building on them.

### Step 1: Review Current Model Architecture
Start by understanding what's already implemented:

- **Current Models**: The project includes `ResNeXtLSTM` (PyTorch) and `EfficientNetGRU` (PyTorch baseline)
- **Current Pipeline**: Uses `train_image_tf.py` (TensorFlow/Keras with MobileNetV2) for image classification
- **What to check**: 
  - See `/src/deepfake_detector/models/` for existing PyTorch models
  - Review `/scripts/train_image_tf.py` for TensorFlow training pipeline
  - Examine `/configs/default.yaml` for current hyperparameters

**Action**: Run this command to verify your setup:
```bash
python3 -c "import torch; import tensorflow as tf; print(f'PyTorch: {torch.__version__}'); print(f'TensorFlow: {tf.__version__}')"
```

### Step 2: Understand the Existing Data Pipeline
The current system handles both video and image inputs:

- **Video Processing**: Extracts frames, detects faces, and feeds sequences to models
- **Image Processing**: Direct classification of still images (2D images → REAL/FAKE)
- **Current Flow**: Frame extraction → Face detection → Preprocessing → Model inference

**Action**: Examine the data handling code:
```bash
cat src/deepfake_detector/data/video.py  # Video frame extraction
cat src/deepfake_detector/data/dataset.py  # Dataset class
```

### Step 3: Set Up Your Development Environment
Prepare your workspace for advanced training:

- **GPU Check**: Ensure you have GPU access (CUDA for PyTorch, CUDA/Metal for TensorFlow)
- **Dependencies**: All required packages are in `requirements.txt`
- **Directory Structure**: Create directories for datasets and models

**Action**: Run these setup commands:
```bash
# Check GPU availability
python -c "import torch; print(f'GPU Available: {torch.cuda.is_available()}')"
python -c "import tensorflow as tf; print(f'GPU Devices: {len(tf.config.list_physical_devices(\"GPU\"))}')"

# Create necessary directories
mkdir -p data/raw/deepfakes_videos
mkdir -p data/processed/faceforensics
mkdir -p models/checkpoints/efficientnet
mkdir -p models/checkpoints/xceptionnet
```

---

## Part 2: Dataset Setup

**What you'll learn:** Download and organize the FaceForensics dataset, which is one of the largest deepfake detection datasets available.

### Step 1: Understanding FaceForensics Dataset
FaceForensics is a comprehensive dataset for deepfake detection research:

- **Size**: ~1000 original videos + multiple compression levels (c0, c23, c40)
- **Manipulations**: Contains 5 different deepfake generation methods (Face2Face, FaceSwap, Deepfakes, NeuralTextures, and re-enactment)
- **Quality Levels**:
  - `c0`: Lossless/highest quality (largest file size)
  - `c23`: Medium compression (balanced)
  - `c40`: High compression (smallest file size, best for training speed)

**Action**: Learn more about the dataset by reviewing its official documentation:
```bash
# Learn the dataset structure
echo "Visit: https://github.com/ondyari/FaceForensics to understand dataset structure"
echo "Download options: https://github.com/ondyari/FaceForensics#download-the-dataset"
```

### Step 2: Download FaceForensics Dataset
There are two ways to download:

**Method A - Using Official Script (Recommended)** (if you have the download credentials):
```bash
# Clone the FaceForensics repo
git clone https://github.com/ondyari/FaceForensics.git
cd FaceForensics

# Run their download script (requires agreeing to terms)
python download-all.py --output_dir ~/data/faceforensics --compression c23 --types raw
```

**Method B - Manual Download**:
- Visit https://github.com/ondyari/FaceForensics#download-the-dataset
- Download the `c23` compression level (balanced quality/size)
- Extract to `data/raw/faceforensics/`

**Action**: Set the downloaded data path:
```bash
# After downloading, verify structure
ls ~/data/faceforensics/  # Should see: original_sequences/, manipulated_sequences/, etc.
```

### Step 3: Organize Dataset for Training
Convert raw video data into frame datasets organized for easy training:

- **Create Frame Directories**: Extract frames from videos into organized folders
- **Label Structure**: Create `train/real/` and `train/fake/` directories
- **Split Dataset**: Allocate 70% training, 15% validation, 15% testing

**Action**: Create a dataset organization script:
```bash
# Create the directory structure
mkdir -p data/processed/faceforensics/{train,validation,test}/{real,fake}

# This preprocessing step should:
# 1. Extract frames from videos
# 2. Detect and crop faces
# 3. Organize into train/val/test splits
# (We'll create this script in Part 4)
```

---

## Part 3: Understanding Advanced Models

**What you'll learn:** Compare EfficientNet and XceptionNet architectures and understand their advantages for deepfake detection.

### Step 1: EfficientNet Architecture Overview
EfficientNet is a family of models that balance accuracy and efficiency:

- **Scaling Method**: Uses compound scaling (depth, width, height) systematically
- **Variants**: B0 to B7 (B0 = smallest/fastest, B7 = largest/most accurate)
- **Why Use It**: 
  - More accurate than ResNet with fewer parameters
  - Efficient scaling makes it practical for production
  - Better transfer learning performance
  - Recommended for mobile and edge deployment

**Key Characteristics**:
```
EfficientNet-B0:  5.3M params, best for real-time inference
EfficientNet-B3:  12M params, good balance
EfficientNet-B5:  30M params, high accuracy, slower
EfficientNet-B7:  66M params, highest accuracy, requires significant GPU
```

**Action**: Test loading an EfficientNet model:
```python
from torchvision import models

# Load pretrained EfficientNet-B3 (good balance for deepfakes)
model = models.efficientnet_b3(weights=models.EfficientNet_B3_Weights.DEFAULT)
print(f"Loaded model with {sum(p.numel() for p in model.parameters())} parameters")
```

### Step 2: XceptionNet Architecture Overview
Xception (Extreme Inception) is another powerful architecture specifically useful for deepfakes:

- **Design**: Separates spatial and channel-wise convolutions (Depthwise Separable Convolutions)
- **Why Use It**: 
  - Originally trained on ImageNet like ResNet/EfficientNet
  - Better feature extraction for fine-grained facial manipulations
  - Fewer parameters than traditional CNNs
  - Many deepfake detection papers use Xception as baseline

**Key Characteristics**:
```
Xception Base: 22.9M params
- Good balance of speed and accuracy
- Excellent for facial feature detection
- Works well with transfer learning
```

**Action**: Test loading an Xception model:
```python
from torchvision import models

# Load pretrained Xception (note: torchvision includes it)
model = models.xception(pretrained=True)
print(f"Loaded Xception with {sum(p.numel() for p in model.parameters())} parameters")
```

### Step 3: Model Comparison and Selection
Compare these models to choose the best fit for your use case:

| Aspect | EfficientNet-B3 | EfficientNet-B5 | Xception |
|--------|-----------------|-----------------|----------|
| Parameters | 12M | 30M | 22.9M |
| Speed | Fast | Medium | Fast |
| Accuracy | High | Very High | High |
| Memory | Low | Medium | Medium |
| Best For | Real-time | Highest accuracy | Balance |

**Action**: Create a model comparison script:
```python
import torch
from torchvision.models import efficientnet_b3, efficientnet_b5, xception

def count_params(model):
    return sum(p.numel() for p in model.parameters())

models_to_test = {
    'EfficientNet-B3': efficientnet_b3(pretrained=True),
    'EfficientNet-B5': efficientnet_b5(pretrained=True),
    'Xception': xception(pretrained=True),
}

for name, model in models_to_test.items():
    print(f"{name}: {count_params(model):,} parameters")
    
    # Time inference on dummy input
    dummy = torch.randn(1, 3, 224, 224)
    if torch.cuda.is_available():
        model = model.cuda()
        dummy = dummy.cuda()
    
    import time
    start = time.time()
    with torch.no_grad():
        _ = model(dummy)
    print(f"  → Inference time: {(time.time() - start)*1000:.2f}ms\n")
```

---

## Part 4: Data Preparation

**What you'll learn:** Extract frames from FaceForensics videos, detect faces, and organize data for optimal training.

### Step 1: Extract Frames from Videos
Convert video files into individual frame images:

- **Why**: Models train on images, not videos. We need to extract frames first
- **Sampling**: Extract every Nth frame (not every frame) to reduce dataset size
- **Resolution**: Keep frames at 224x224 or 256x256 for faster processing

**Action**: Create frame extraction script (`scripts/extract_frames.py`):
```python
import cv2
import os
from pathlib import Path
from tqdm import tqdm

def extract_frames(video_path, output_dir, frame_skip=5, target_size=224):
    """Extract frames from video and save as images"""
    cap = cv2.VideoCapture(str(video_path))
    frame_count = 0
    saved_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Only extract every Nth frame
        if frame_count % frame_skip == 0:
            # Resize frame
            resized = cv2.resize(frame, (target_size, target_size))
            
            # Save frame
            output_file = output_dir / f"frame_{saved_count:06d}.jpg"
            cv2.imwrite(str(output_file), resized)
            saved_count += 1
        
        frame_count += 1
    
    cap.release()
    return saved_count

# Usage example:
video_path = "data/raw/faceforensics/original_sequences/video/c23/001_0_0.mp4"
output_dir = Path("data/processed/faceforensics/train/real")
output_dir.mkdir(parents=True, exist_ok=True)

frames_extracted = extract_frames(video_path, output_dir)
print(f"Extracted {frames_extracted} frames")
```

### Step 2: Detect and Crop Faces
Extract only face regions from frames for focused learning:

- **Why**: Facial regions are most informative for deepfake detection
- **Tool**: Use MTCNN or RetinaFace for reliable face detection
- **Benefit**: Reduces irrelevant background information, improves model focus

**Action**: Create face detection script (`scripts/detect_faces.py`):
```python
import cv2
import torch
from facenet_pytorch import MTCNN
from pathlib import Path
import numpy as np

def detect_and_crop_faces(image_path, output_dir, confidence_threshold=0.9):
    """Detect faces in image and save cropped faces"""
    
    # Initialize MTCNN (Fast CNN-based face detector)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    mtcnn = MTCNN(device=device, thresholds=[0.6, 0.7, 0.7])
    
    img = cv2.imread(str(image_path))
    if img is None:
        return 0
    
    # Convert BGR to RGB
    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Detect faces
    try:
        boxes, probs = mtcnn.detect(rgb_img)
        if boxes is None:
            return 0
    except:
        return 0
    
    saved_count = 0
    for idx, (box, prob) in enumerate(zip(boxes, probs)):
        if prob < confidence_threshold:
            continue
        
        # Extract face region
        x1, y1, x2, y2 = [int(coord) for coord in box]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(img.shape[1], x2), min(img.shape[0], y2)
        
        face = img[y1:y2, x1:x2]
        if face.shape[0] > 10 and face.shape[1] > 10:  # Minimum size check
            output_file = output_dir / f"{image_path.stem}_face_{idx}.jpg"
            cv2.imwrite(str(output_file), face)
            saved_count += 1
    
    return saved_count

# Usage:
input_dir = Path("data/processed/faceforensics/train/real")
output_dir = Path("data/processed/faceforensics_faces/train/real")
output_dir.mkdir(parents=True, exist_ok=True)

for image_file in sorted(input_dir.glob("*.jpg"))[:100]:  # Process first 100
    detect_and_crop_faces(image_file, output_dir)
```

### Step 3: Create Training Data Splits
Organize data into training, validation, and test sets:

- **70-15-15 Split**: 70% training, 15% validation, 15% testing
- **Stratification**: Ensure each split has balanced REAL/FAKE samples
- **Reproducibility**: Use random seeds for consistency

**Action**: Create data split script (`scripts/create_splits.py`):
```python
import os
import shutil
from pathlib import Path
from sklearn.model_selection import train_test_split
import numpy as np

def create_train_val_test_splits(source_dir, output_dir, val_size=0.15, test_size=0.15):
    """
    Split dataset into train/val/test while maintaining class balance
    
    Expected source_dir structure:
    source_dir/
        real/
            *.jpg
        fake/
            *.jpg
    """
    
    output_path = Path(output_dir)
    
    # Create output directories
    for split in ['train', 'validation', 'test']:
        for label in ['real', 'fake']:
            (output_path / split / label).mkdir(parents=True, exist_ok=True)
    
    # Process each class
    for label in ['real', 'fake']:
        label_dir = Path(source_dir) / label
        images = list(label_dir.glob("*.jpg"))
        
        # First split: train vs others (85% vs 15%)
        train_imgs, others = train_test_split(
            images, 
            test_size=(val_size + test_size), 
            random_state=42
        )
        
        # Second split: val vs test (50% vs 50% of others)
        val_imgs, test_imgs = train_test_split(
            others, 
            test_size=0.5, 
            random_state=42
        )
        
        # Copy files to appropriate directories
        for img in train_imgs:
            shutil.copy(img, output_path / 'train' / label / img.name)
        for img in val_imgs:
            shutil.copy(img, output_path / 'validation' / label / img.name)
        for img in test_imgs:
            shutil.copy(img, output_path / 'test' / label / img.name)
        
        print(f"Label '{label}':")
        print(f"  Train: {len(train_imgs)}")
        print(f"  Val: {len(val_imgs)}")
        print(f"  Test: {len(test_imgs)}")

# Usage:
create_train_val_test_splits(
    'data/processed/faceforensics_faces/raw',
    'data/processed/faceforensics_faces/split'
)
```

---

## Part 5: Building Training Infrastructure

**What you'll learn:** Set up the PyTorch training pipeline for EfficientNet and XceptionNet with proper config management.

### Step 1: Create PyTorch Model Wrappers
Wrap EfficientNet and XceptionNet for consistent training:

- **Standardization**: Create a unified interface for both models
- **Fine-tuning Layers**: Implement layer freezing strategies for transfer learning
- **Output Adaptation**: Transform model outputs for binary classification (REAL/FAKE)

**Action**: Create `src/deepfake_detector/models/efficientnet_classifier.py`:
```python
import torch
import torch.nn as nn
from torchvision import models

class EfficientNetClassifier(nn.Module):
    """
    EfficientNet-based image classifier for REAL vs FAKE deepfakes
    Supports B0-B7 variants with configurable fine-tuning
    """
    
    def __init__(self, variant='b3', num_classes=2, pretrained=True, freeze_backbone=False):
        super().__init__()
        
        # Load EfficientNet variant
        model_dict = {
            'b0': (models.efficientnet_b0, models.EfficientNet_B0_Weights.DEFAULT),
            'b3': (models.efficientnet_b3, models.EfficientNet_B3_Weights.DEFAULT),
            'b5': (models.efficientnet_b5, models.EfficientNet_B5_Weights.DEFAULT),
            'b7': (models.efficientnet_b7, models.EfficientNet_B7_Weights.DEFAULT),
        }
        
        if variant not in model_dict:
            raise ValueError(f"Unknown variant: {variant}. Choose from {list(model_dict.keys())}")
        
        model_fn, weights = model_dict[variant]
        self.backbone = model_fn(weights=weights if pretrained else None)
        
        # Replace classification head
        num_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(num_features, num_classes)
        )
        
        # Optionally freeze backbone
        if freeze_backbone:
            for param in self.backbone.features.parameters():
                param.requires_grad = False
    
    def forward(self, x):
        return self.backbone(x)
    
    def unfreeze_backbone(self, num_layers=None):
        """Unfreeze backbone for fine-tuning"""
        if num_layers is None:
            # Unfreeze all
            for param in self.backbone.features.parameters():
                param.requires_grad = True
        else:
            # Unfreeze only top N layers
            total_layers = len(list(self.backbone.features.parameters()))
            for idx, param in enumerate(self.backbone.features.parameters()):
                if idx >= (total_layers - num_layers):
                    param.requires_grad = True

# Test it:
model = EfficientNetClassifier(variant='b3', num_classes=2, pretrained=True)
dummy_input = torch.randn(2, 3, 224, 224)
output = model(dummy_input)
print(f"Model output shape: {output.shape}")  # Should be [2, 2]
```

### Step 2: Create XceptionNet Wrapper
Similarly, wrap XceptionNet for deepfake classification:

- **Consistency**: Match the interface of EfficientNet wrapper
- **Layer Adaptation**: Modify final layers for binary classification
- **Flexibility**: Allow easy freezing/unfreezing of layers

**Action**: Create `src/deepfake_detector/models/xceptionnet_classifier.py`:
```python
import torch
import torch.nn as nn
from torchvision import models

class XceptionNetClassifier(nn.Module):
    """
    Xception-based image classifier for REAL vs FAKE deepfakes
    Good for detailed facial feature detection
    """
    
    def __init__(self, num_classes=2, pretrained=True, freeze_backbone=False):
        super().__init__()
        
        self.backbone = models.xception(
            weights=models.Xception_Weights.DEFAULT if pretrained else None
        )
        
        # Replace final classification layer
        num_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(num_features, num_classes)
        )
        
        if freeze_backbone:
            # Freeze all conv layers
            for param in self.backbone.conv1.parameters():
                param.requires_grad = False
            for param in self.backbone.conv2.parameters():
                param.requires_grad = False
            for param in self.backbone.block1.parameters():
                param.requires_grad = False
            for param in self.backbone.block2.parameters():
                param.requires_grad = False
            for param in self.backbone.block3.parameters():
                param.requires_grad = False
            for param in self.backbone.block4.parameters():
                param.requires_grad = False
            for param in self.backbone.block5.parameters():
                param.requires_grad = False
            for param in self.backbone.block6.parameters():
                param.requires_grad = False
            for param in self.backbone.block7.parameters():
                param.requires_grad = False
            for param in self.backbone.block8.parameters():
                param.requires_grad = False
            for param in self.backbone.block9.parameters():
                param.requires_grad = False
            for param in self.backbone.block10.parameters():
                param.requires_grad = False
            for param in self.backbone.block11.parameters():
                param.requires_grad = False
            for param in self.backbone.block12.parameters():
                param.requires_grad = False
    
    def forward(self, x):
        return self.backbone(x)
    
    def unfreeze_backbone(self, num_blocks=None):
        """Unfreeze backbone for fine-tuning"""
        # This unfreezes top N blocks
        if num_blocks is None:
            # Unfreeze all
            for param in self.backbone.parameters():
                if "fc" not in param:
                    param.requires_grad = True

# Test it:
model = XceptionNetClassifier(num_classes=2, pretrained=True)
dummy_input = torch.randn(2, 3, 224, 224)
output = model(dummy_input)
print(f"Model output shape: {output.shape}")  # Should be [2, 2]
```

### Step 3: Create Training Configuration
Set up a YAML configuration for easy hyperparameter management:

- **Reproducibility**: Store all hyperparameters in config files
- **Flexibility**: Easy to switch between models and datasets
- **Versioning**: Track which config was used for which model

**Action**: Create `configs/advanced_training.yaml`:
```yaml
# Advanced Training Configuration for EfficientNet and XceptionNet

# Dataset Configuration
dataset:
  name: "faceforensics_frames"
  base_dir: "data/processed/faceforensics_faces/split"
  image_size: 224
  batch_size: 32
  num_workers: 4

# Model Configuration
model:
  type: "efficientnet"  # Options: efficientnet, xceptionnet
  variant: "b3"  # For EfficientNet: b0, b3, b5, b7
  num_classes: 2
  pretrained: true
  freeze_backbone_initial: true  # Freeze before unfreezing top layers

# Training Configuration - Stage 1 (Warm-up)
training:
  stage1:
    epochs: 10
    learning_rate: 1e-4
    optimizer: "adam"
    scheduler: "cosine"  # Options: constant, cosine, exponential
    warmup_epochs: 2
    class_weights: true  # Handle class imbalance
    
  # Stage 2 (Fine-tuning)
  stage2:
    epochs: 10
    learning_rate: 1e-5
    optimizer: "adam"
    scheduler: "cosine"
    unfreeze_layers: 20  # Number of layers to unfreeze

# Augmentation Configuration
augmentation:
  train:
    random_flip: true
    random_rotation: 15
    color_jitter:
      brightness: 0.2
      contrast: 0.2
      saturation: 0.2
      hue: 0.1
    random_affine: true
    random_perspective: true
    gaussian_blur: 0.5  # Probability
    cutout: true
    cutout_size: 32
    
  validation:
    random_flip: false
    random_rotation: 0

# Checkpointing
checkpoint:
  save_dir: "models/checkpoints/advanced"
  save_frequency: 1  # Save every N epochs
  keep_best: true
  metric: "val_auc"  # Monitor metric for best model

# Logging
logging:
  log_dir: "logs/advanced"
  tensorboard: true
  wandb: false  # Set to true if using Weights & Biases
```

---

## Part 6: Advanced Augmentation

**What you'll learn:** Implement sophisticated data augmentation techniques specific to deepfake detection.

### Step 1: Implement Advanced Image Augmentation
Use deeper augmentation strategies to improve model robustness:

- **Why**: Augmentation prevents overfitting and makes models robust to compression/artifacts
- **Deepfake-Specific**: Include compression artifacts, color shifts (common in deepfakes)
- **Balance**: Don't over-augment; maintain image realism

**Action**: Create `src/deepfake_detector/data/augmentation.py`:
```python
import torchvision.transforms as transforms
from torchvision.transforms import v2
import torch
import numpy as np

class DeepfakeAugmentationPipeline:
    """
    Advanced augmentation for deepfake detection
    Includes realistic degradation patterns found in fake videos
    """
    
    @staticmethod
    def get_train_transforms(image_size=224):
        """Training augmentation with deepfake-specific techniques"""
        return v2.Compose([
            v2.Resize((image_size, image_size)),
            
            # Geometric augmentations
            v2.RandomHorizontalFlip(p=0.5),
            v2.RandomVerticalFlip(p=0.1),
            v2.RandomRotation(degrees=15),
            v2.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
            v2.RandomPerspective(distortion_scale=0.2, p=0.5),
            
            # Color augmentations (compression artifacts appear as color distortions)
            v2.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.15),
            
            # Blur (simulates compression)
            v2.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
            
            # Random erasing (cutout) to prevent memorization
            v2.RandomErasing(p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.0)),
            
            # Normalization
            v2.ToTensor(),
            v2.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    
    @staticmethod
    def get_val_transforms(image_size=224):
        """Validation augmentation (minimal, deterministic)"""
        return v2.Compose([
            v2.Resize((image_size, image_size)),
            v2.ToTensor(),
            v2.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    
    @staticmethod
    def get_test_transforms(image_size=224):
        """Test augmentation (no randomization, TTA ready)"""
        return v2.Compose([
            v2.Resize((image_size, image_size)),
            v2.ToTensor(),
            v2.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

# Usage:
train_transform = DeepfakeAugmentationPipeline.get_train_transforms(224)
val_transform = DeepfakeAugmentationPipeline.get_val_transforms(224)

# Apply to dataset:
# from torchvision.datasets import ImageFolder
# train_dataset = ImageFolder(
#     root='data/train',
#     transform=train_transform
# )
```

### Step 2: Add Test-Time Augmentation (TTA)
Use multiple augmented versions of test images to improve predictions:

- **Why**: TTA improves robustness by averaging predictions across variations
- **How**: Apply different augmentations to test image, average predictions
- **Cost**: Slower inference, but more reliable results

**Action**: Create TTA utility in `src/deepfake_detector/utils/tta.py`:
```python
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms

class TestTimeAugmentation:
    """
    Test-Time Augmentation for improved robustness
    """
    
    def __init__(self, num_augmentations=5):
        self.num_augmentations = num_augmentations
        self.augmentations = [
            transforms.Compose([
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=10),
            ]),
            transforms.Compose([
                transforms.ColorJitter(brightness=0.1, contrast=0.1),
            ]),
            transforms.Compose([
                transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),
            ]),
            transforms.Compose([
                transforms.GaussianBlur(kernel_size=3),
            ]),
            transforms.Compose([
                transforms.RandomErasing(p=0.3),
            ])
        ]
    
    @torch.no_grad()
    def predict_with_tta(self, model, image, device):
        """
        Make predictions using Test-Time Augmentation
        
        Args:
            model: PyTorch model
            image: Input image tensor [C, H, W]
            device: torch.device
        
        Returns:
            averaged predictions and confidence
        """
        
        predictions = []
        
        # Original image
        image_batch = image.unsqueeze(0).to(device)
        with torch.no_grad():
            output = model(image_batch)
            predictions.append(F.softmax(output, dim=1))
        
        # Augmented versions
        for aug in self.augmentations:
            try:
                aug_image = aug(image)
                image_batch = aug_image.unsqueeze(0).to(device)
                output = model(image_batch)
                predictions.append(F.softmax(output, dim=1))
            except:
                pass  # Skip if augmentation fails
        
        # Average predictions
        avg_prediction = torch.mean(torch.cat(predictions, dim=0), dim=0)
        confidence, predicted_class = torch.max(avg_prediction, dim=0)
        
        return int(predicted_class.item()), float(confidence.item())

# Usage:
# tta = TestTimeAugmentation(num_augmentations=5)
# pred_class, confidence = tta.predict_with_tta(model, test_image, device)
```

### Step 3: Implement Mixup and CutMix Augmentation
Advanced augmentation techniques that mix multiple samples:

- **Mixup**: Blend two images and their labels
- **CutMix**: Replace a rectangular patch with another image's patch
- **Benefit**: Improves generalization and robustness

**Action**: Add to `src/deepfake_detector/data/augmentation.py`:
```python
import torch
import torch.nn.functional as F

class MixupAugmentation:
    """Mixup regularization technique"""
    
    def __init__(self, alpha=1.0):
        self.alpha = alpha
    
    def __call__(self, batch_images, batch_labels):
        """
        Apply mixup to a batch
        
        Args:
            batch_images: [B, C, H, W]
            batch_labels: [B, num_classes] or [B]
        
        Returns:
            mixed_images, mixed_labels
        """
        lam = np.random.beta(self.alpha, self.alpha)
        batch_size = batch_images.size(0)
        
        index = torch.randperm(batch_size)
        
        mixed_images = lam * batch_images + (1 - lam) * batch_images[index]
        
        if batch_labels.dim() == 1:
            # Convert to one-hot
            num_classes = batch_labels.max().int().item() + 1
            mixed_labels = torch.zeros(batch_size, num_classes)
            mixed_labels.scatter_(1, batch_labels.unsqueeze(1), 1.0)
            target_labels = torch.zeros(batch_size, num_classes)
            target_labels.scatter_(1, batch_labels[index].unsqueeze(1), 1.0)
        else:
            mixed_labels = batch_labels
            target_labels = batch_labels[index]
        
        mixed_labels = lam * mixed_labels + (1 - lam) * target_labels
        
        return mixed_images, mixed_labels

class CutmixAugmentation:
    """CutMix augmentation"""
    
    def __init__(self, alpha=1.0):
        self.alpha = alpha
    
    def __call__(self, batch_images, batch_labels):
        """Apply cutmix to batch"""
        lam = np.random.beta(self.alpha, self.alpha)
        batch_size = batch_images.size(0)
        _, _, height, width = batch_images.shape
        
        index = torch.randperm(batch_size)
        
        # Sample box
        cut_ratio = np.sqrt(1.0 - lam)
        cut_h = int(height * cut_ratio)
        cut_w = int(width * cut_ratio)
        
        cx = np.random.randint(0, width)
        cy = np.random.randint(0, height)
        
        bbx1 = np.clip(cx - cut_w // 2, 0, width)
        bby1 = np.clip(cy - cut_h // 2, 0, height)
        bbx2 = np.clip(cx + cut_w // 2, 0, width)
        bby2 = np.clip(cy + cut_h // 2, 0, height)
        
        mixed_images = batch_images.clone()
        mixed_images[:, :, bby1:bby2, bbx1:bbx2] = batch_images[index, :, bby1:bby2, bbx1:bbx2]
        
        # Adjust lambda
        lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1)) / (height * width)
        
        # Mix labels
        if batch_labels.dim() == 1:
            num_classes = batch_labels.max().int().item() + 1
            mixed_labels = torch.zeros(batch_size, num_classes)
            mixed_labels.scatter_(1, batch_labels.unsqueeze(1), lam)
            mixed_labels.scatter_(1, batch_labels[index].unsqueeze(1), 1 - lam, reduce='add')
        else:
            mixed_labels = lam * batch_labels + (1 - lam) * batch_labels[index]
        
        return mixed_images, mixed_labels, lam

# Usage in training loop:
# mixup = MixupAugmentation(alpha=1.0)
# for images, labels in train_loader:
#     images, labels = mixup(images, labels)
#     # ... training step
```

---

## Part 7: Transfer Learning & Fine-tuning

**What you'll learn:** Implement a multi-stage training strategy that leverages pretrained ImageNet weights.

### Step 1: Understand Transfer Learning Principles
Learn why transfer learning works for deepfake detection:

- **ImageNet Pretraining**: Models pretrained on ImageNet have learned general features (edges, textures, shapes)
- **Two Stages**: 
  - Stage 1: Train only classification head (backbone frozen)
  - Stage 2: Fine-tune backbone layers with lower learning rate
- **Why It Works**: Deepfakes still have faces and facial features similar to real faces; general feature detection helps

**Concept Summary**:
```
Stage 1: Frozen Backbone → Train Head
├─ Backbone: Keep pretrained ImageNet weights (frozen)
├─ Head: Train classification layers from scratch
├─ Learning Rate: 1e-4 (moderate)
└─ Duration: 10 epochs

Stage 2: Unfreeze Top Layers → Fine-tune
├─ Backbone: Unfreeze top 20-50 layers
├─ Head: Continue training
├─ Learning Rate: 1e-5 (very low, careful adjustment)
└─ Duration: 10 epochs
```

**Action**: Study the principle with a simple example:
```python
# Check which layers are trainable
model = EfficientNetClassifier(variant='b3', freeze_backbone=True)

print("Stage 1 - Frozen Backbone:")
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
total_params = sum(p.numel() for p in model.parameters())
print(f"Trainable: {trainable_params:,} / Total: {total_params:,}")
print(f"Trainable %: {trainable_params/total_params*100:.1f}%")

# Unfreeze for Stage 2
model.unfreeze_backbone(num_layers=20)
print("\nStage 2 - Unfroze Top 20 Layers:")
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Trainable: {trainable_params:,} / Total: {total_params:,}")
print(f"Trainable %: {trainable_params/total_params*100:.1f}%")
```

### Step 2: Implement Multi-Stage Training Loop
Create a training function that handles both stages:

- **Stage 1**: Quick convergence of classification head
- **Stage 2**: Fine-tuning backbone for task-specific features
- **Monitoring**: Track metrics separately for each stage

**Action**: Create `src/deepfake_detector/pipelines/multi_stage_trainer.py`:
```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
from tqdm import tqdm
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

class MultiStageTrainer:
    """
    Multi-stage training for transfer learning
    Stage 1: Train classification head (frozen backbone)
    Stage 2: Fine-tune backbone layers
    """
    
    def __init__(self, model, device, num_classes=2):
        self.model = model
        self.device = device
        self.num_classes = num_classes
        self.criterion = nn.CrossEntropyLoss()
    
    def train_epoch(self, train_loader, optimizer, epoch, num_epochs):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        all_preds = []
        all_targets = []
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        
        for batch_idx, (images, labels) in enumerate(pbar):
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            optimizer.zero_grad()
            
            # Forward pass
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_loss += loss.item()
            
            # Compute accuracy
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(labels.cpu().numpy())
            
            pbar.set_postfix({
                'loss': f'{total_loss / (batch_idx + 1):.4f}',
                'accuracy': f'{accuracy_score(all_targets, all_preds):.4f}'
            })
        
        return total_loss / len(train_loader), accuracy_score(all_targets, all_preds)
    
    @torch.no_grad()
    def validate(self, val_loader):
        """Validate model"""
        self.model.eval()
        total_loss = 0.0
        all_preds = []
        all_probs = []
        all_targets = []
        
        for images, labels in tqdm(val_loader, desc="Validating"):
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            
            total_loss += loss.item()
            
            probs = torch.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            all_targets.extend(labels.cpu().numpy())
        
        accuracy = accuracy_score(all_targets, all_preds)
        f1 = f1_score(all_targets, all_preds, average='weighted')
        auc = roc_auc_score(all_targets, np.array(all_probs)[:, 1])
        
        return total_loss / len(val_loader), accuracy, f1, auc
    
    def train_stage1(self, train_loader, val_loader, epochs=10, lr=1e-4):
        """
        Stage 1: Train classification head only (frozen backbone)
        """
        print("\n" + "="*60)
        print("STAGE 1: Training Classification Head")
        print("="*60)
        
        optimizer = optim.Adam(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=lr
        )
        scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
        
        best_auc = 0.0
        best_weights = None
        
        for epoch in range(epochs):
            train_loss, train_acc = self.train_epoch(train_loader, optimizer, epoch, epochs)
            val_loss, val_acc, val_f1, val_auc = self.validate(val_loader)
            
            print(f"\nEpoch {epoch+1}/{epochs}")
            print(f"  Train - Loss: {train_loss:.4f}, Accuracy: {train_acc:.4f}")
            print(f"  Val   - Loss: {val_loss:.4f}, Accuracy: {val_acc:.4f}, F1: {val_f1:.4f}, AUC: {val_auc:.4f}")
            
            scheduler.step()
            
            # Save best model
            if val_auc > best_auc:
                best_auc = val_auc
                best_weights = {k: v.cpu() for k, v in self.model.state_dict().items()}
            
            if (epoch + 1) % 5 == 0:
                print(f"  → Best AUC so far: {best_auc:.4f}")
        
        # Load best weights
        if best_weights:
            self.model.load_state_dict(best_weights)
            print(f"\nLoaded best model (AUC: {best_auc:.4f})")
    
    def train_stage2(self, train_loader, val_loader, epochs=10, lr=1e-5, unfreeze_layers=20):
        """
        Stage 2: Fine-tune backbone layers
        """
        print("\n" + "="*60)
        print("STAGE 2: Fine-tuning Backbone Layers")
        print("="*60)
        
        # Unfreeze the backbone
        self.model.unfreeze_backbone(num_layers=unfreeze_layers)
        
        # Verify some layers are now trainable
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.model.parameters())
        print(f"Trainable parameters: {trainable:,} / {total:,} ({trainable/total*100:.1f}%)\n")
        
        optimizer = optim.Adam(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=lr
        )
        scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2, verbose=True)
        
        best_auc = 0.0
        best_weights = None
        
        for epoch in range(epochs):
            train_loss, train_acc = self.train_epoch(train_loader, optimizer, epoch, epochs)
            val_loss, val_acc, val_f1, val_auc = self.validate(val_loader)
            
            print(f"\nEpoch {epoch+1}/{epochs}")
            print(f"  Train - Loss: {train_loss:.4f}, Accuracy: {train_acc:.4f}")
            print(f"  Val   - Loss: {val_loss:.4f}, Accuracy: {val_acc:.4f}, F1: {val_f1:.4f}, AUC: {val_auc:.4f}")
            
            scheduler.step(val_auc)
            
            if val_auc > best_auc:
                best_auc = val_auc
                best_weights = {k: v.cpu() for k, v in self.model.state_dict().items()}
        
        if best_weights:
            self.model.load_state_dict(best_weights)
            print(f"\nLoaded best model (AUC: {best_auc:.4f})")

# Usage:
# trainer = MultiStageTrainer(model, device='cuda')
# trainer.train_stage1(train_loader, val_loader, epochs=10, lr=1e-4)
# trainer.train_stage2(train_loader, val_loader, epochs=10, lr=1e-5, unfreeze_layers=20)
```

### Step 3: Implement Learning Rate Scheduling
Adjust learning rate during training for better convergence:

- **Why**: High LR early helps escape local minima; low LR later fine-tunes weights
- **Strategies**: Cosine annealing (smooth decrease), ReduceLROnPlateau (adaptive)
- **Stage-Specific**: Use different schedules for each stage

**Action**: The scheduler is already in the above code, but here's a detailed explanation:
```python
# Cosine Annealing (smooth decay)
scheduler = CosineAnnealingLR(
    optimizer,
    T_max=epochs,  # Number of epochs
    eta_min=1e-7   # Minimum learning rate
)

# ReduceLROnPlateau (adaptive - reduce when metric plateaus)
scheduler = ReduceLROnPlateau(
    optimizer,
    mode='max',      # Maximize validation AUC
    factor=0.5,      # Multiply LR by 0.5
    patience=2,      # Wait 2 epochs before reducing
    verbose=True
)

# Warmup + Cosine (advanced)
class WarmupCosineScheduler:
    def __init__(self, optimizer, warmup_epochs, total_epochs):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
    
    def step(self, epoch):
        if epoch < self.warmup_epochs:
            # Linear warmup
            lr = 1e-5 + (1e-4 - 1e-5) * epoch / self.warmup_epochs
        else:
            # Cosine annealing
            progress = (epoch - self.warmup_epochs) / (self.total_epochs - self.warmup_epochs)
            lr = 1e-5 + 0.5 * (1e-4 - 1e-5) * (1 + np.cos(np.pi * progress))
        
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
```

---

## Part 8: Training & Monitoring

**What you'll learn:** Set up end-to-end training with proper monitoring, checkpointing, and early stopping.

### Step 1: Create Training Script
Build the main training script combining all components:

- **Configuration Loading**: Read from YAML config file
- **Data Loading**: Create DataLoaders with augmentation
- **Training Loop**: Call multi-stage trainer
- **Checkpointing**: Save models at regular intervals

**Action**: Create `scripts/train_advanced.py`:
```python
import argparse
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from deepfake_detector.models.efficientnet_classifier import EfficientNetClassifier
from deepfake_detector.models.xceptionnet_classifier import XceptionNetClassifier
from deepfake_detector.data.augmentation import DeepfakeAugmentationPipeline
from deepfake_detector.pipelines.multi_stage_trainer import MultiStageTrainer

def load_config(config_path):
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def create_dataloaders(config):
    """Create train and validation dataloaders"""
    base_dir = Path(config['dataset']['base_dir'])
    
    # Create augmentation pipelines
    train_transform = DeepfakeAugmentationPipeline.get_train_transforms(
        config['dataset']['image_size']
    )
    val_transform = DeepfakeAugmentationPipeline.get_val_transforms(
        config['dataset']['image_size']
    )
    
    # Create datasets
    train_dataset = ImageFolder(
        root=str(base_dir / 'train'),
        transform=train_transform
    )
    
    val_dataset = ImageFolder(
        root=str(base_dir / 'validation'),
        transform=val_transform
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['dataset']['batch_size'],
        shuffle=True,
        num_workers=config['dataset']['num_workers'],
        pin_memory=torch.cuda.is_available()
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['dataset']['batch_size'],
        shuffle=False,
        num_workers=config['dataset']['num_workers'],
        pin_memory=torch.cuda.is_available()
    )
    
    return train_loader, val_loader

def create_model(config):
    """Create model based on config"""
    model_type = config['model']['type']
    
    if model_type == 'efficientnet':
        model = EfficientNetClassifier(
            variant=config['model']['variant'],
            num_classes=config['model']['num_classes'],
            pretrained=config['model']['pretrained'],
            freeze_backbone=config['model']['freeze_backbone_initial']
        )
    elif model_type == 'xceptionnet':
        model = XceptionNetClassifier(
            num_classes=config['model']['num_classes'],
            pretrained=config['model']['pretrained'],
            freeze_backbone=config['model']['freeze_backbone_initial']
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    return model

def main():
    parser = argparse.ArgumentParser(description="Train advanced deepfake detection models")
    parser.add_argument('--config', type=str, required=True, help='Config file path')
    parser.add_argument('--output-dir', type=str, default='models/checkpoints/advanced', help='Output directory')
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create dataloaders
    print("Creating dataloaders...")
    train_loader, val_loader = create_dataloaders(config)
    print(f"  Train samples: {len(train_loader.dataset)}")
    print(f"  Val samples: {len(val_loader.dataset)}")
    
    # Create model
    print(f"\nCreating model: {config['model']['type']}...")
    model = create_model(config).to(device)
    print(f"Model created successfully")
    
    # Create trainer
    trainer = MultiStageTrainer(model, device, num_classes=config['model']['num_classes'])
    
    # Stage 1 training
    print("\n" + "="*80)
    print("STAGE 1: TRAINING")
    print("="*80)
    trainer.train_stage1(
        train_loader,
        val_loader,
        epochs=config['training']['stage1']['epochs'],
        lr=config['training']['stage1']['learning_rate']
    )
    
    # Save Stage 1 checkpoint
    checkpoint_path = output_dir / "model_stage1.pt"
    torch.save(model.state_dict(), checkpoint_path)
    print(f"Saved Stage 1 checkpoint: {checkpoint_path}")
    
    # Stage 2 training
    print("\n" + "="*80)
    print("STAGE 2: FINE-TUNING")
    print("="*80)
    trainer.train_stage2(
        train_loader,
        val_loader,
        epochs=config['training']['stage2']['epochs'],
        lr=config['training']['stage2']['learning_rate'],
        unfreeze_layers=config['training']['stage2']['unfreeze_layers']
    )
    
    # Save final model
    final_path = output_dir / f"model_final_{config['model']['type']}.pt"
    torch.save(model.state_dict(), final_path)
    print(f"\nSaved final model: {final_path}")
    
    # Also save as ONNX for deployment
    print("\nExporting model to ONNX format...")
    dummy_input = torch.randn(1, 3, config['dataset']['image_size'], config['dataset']['image_size'])
    onnx_path = output_dir / f"model_final_{config['model']['type']}.onnx"
    torch.onnx.export(model, dummy_input.to(device), str(onnx_path), verbose=False)
    print(f"Saved ONNX model: {onnx_path}")

if __name__ == '__main__':
    main()
```

### Step 2: Add Experiment Tracking
Monitor training with metrics logging and visualization:

- **Metrics to Track**: Loss, accuracy, F1, AUC, precision, recall
- **Tools**: TensorBoard (built-in) or Weights & Biases (cloud)
- **Visualization**: Plot training curves, confusion matrices

**Action**: Create `src/deepfake_detector/utils/experiment_tracker.py`:
```python
import json
from pathlib import Path
from datetime import datetime
import numpy as np

try:
    from torch.utils.tensorboard import SummaryWriter
    HAS_TENSORBOARD = True
except ImportError:
    HAS_TENSORBOARD = False

class ExperimentTracker:
    """Track experiment metrics and results"""
    
    def __init__(self, log_dir='./logs', experiment_name=None):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        if experiment_name is None:
            experiment_name = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        self.experiment_dir = self.log_dir / experiment_name
        self.experiment_dir.mkdir(parents=True, exist_ok=True)
        
        # TensorBoard writer
        if HAS_TENSORBOARD:
            self.writer = SummaryWriter(str(self.experiment_dir))
        else:
            self.writer = None
        
        # Metrics storage
        self.metrics = {
            'train': [],
            'val': []
        }
        
        print(f"Tracking experiment: {experiment_name}")
        print(f"Log directory: {self.experiment_dir}")
    
    def log_epoch(self, epoch, train_metrics, val_metrics):
        """Log metrics for an epoch"""
        
        # Store metrics
        self.metrics['train'].append(train_metrics)
        self.metrics['val'].append(val_metrics)
        
        # Log to TensorBoard
        if self.writer:
            for key, value in train_metrics.items():
                self.writer.add_scalar(f'train/{key}', value, epoch)
            for key, value in val_metrics.items():
                self.writer.add_scalar(f'val/{key}', value, epoch)
            self.writer.flush()
        
        # Print summary
        print(f"\nEpoch {epoch + 1}")
        print(f"  Train: {' | '.join(f'{k}={v:.4f}' for k, v in train_metrics.items())}")
        print(f"  Val:   {' | '.join(f'{k}={v:.4f}' for k, v in val_metrics.items())}")
    
    def save_metrics(self):
        """Save metrics to JSON"""
        metrics_path = self.experiment_dir / 'metrics.json'
        with open(metrics_path, 'w') as f:
            json.dump(self.metrics, f, indent=2)
        print(f"Saved metrics: {metrics_path}")
    
    def close(self):
        """Close experiment"""
        if self.writer:
            self.writer.close()
            print("Closed TensorBoard writer")

# Usage:
# tracker = ExperimentTracker(experiment_name='efficientnet_v1')
# for epoch in range(num_epochs):
#     train_metrics = {'loss': 0.5, 'accuracy': 0.92}
#     val_metrics = {'loss': 0.6, 'accuracy': 0.90, 'auc': 0.95}
#     tracker.log_epoch(epoch, train_metrics, val_metrics)
# tracker.save_metrics()
# tracker.close()
```

### Step 3: Implement Early Stopping and Checkpointing
Save best models automatically and stop training when performance plateaus:

- **Early Stopping**: Stop if validation metric doesn't improve for N epochs
- **Best Model Checkpoint**: Save model with best validation AUC/F1
- **Periodic Checkpoints**: Save snapshots at regular intervals

**Action**: Already partially implemented in MultiStageTrainer, but add comprehensive version:
```python
class CheckpointManager:
    """Manage model checkpoints"""
    
    def __init__(self, checkpoint_dir, metric_name='auc', mode='max'):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.metric_name = metric_name
        self.mode = mode  # 'max' for maximization, 'min' for minimization
        self.best_metric = float('-inf') if mode == 'max' else float('inf')
        self.best_checkpoint_path = None
    
    def save_checkpoint(self, model, epoch, metrics):
        """Save checkpoint if it's the best so far"""
        metric_value = metrics.get(self.metric_name)
        if metric_value is None:
            return False
        
        # Check if this is the best
        is_best = False
        if self.mode == 'max' and metric_value > self.best_metric:
            is_best = True
            self.best_metric = metric_value
        elif self.mode == 'min' and metric_value < self.best_metric:
            is_best = True
            self.best_metric = metric_value
        
        # Save checkpoint
        if is_best:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'metrics': metrics
            }
            path = self.checkpoint_dir / 'best_model.pt'
            torch.save(checkpoint, path)
            self.best_checkpoint_path = path
            print(f"  ✓ Saved best model ({self.metric_name}={metric_value:.4f})")
            return True
        
        return False
    
    def load_best_checkpoint(self, model):
        """Load the best checkpoint"""
        if self.best_checkpoint_path and self.best_checkpoint_path.exists():
            checkpoint = torch.load(self.best_checkpoint_path)
            model.load_state_dict(checkpoint['model_state_dict'])
            return checkpoint['epoch'], checkpoint['metrics']
        return None, None

# Usage in training:
# checkpoint_manager = CheckpointManager('checkpoints', metric_name='auc', mode='max')
# for epoch in range(num_epochs):
#     train_loss = train_one_epoch()
#     val_metrics = validate()
#     checkpoint_manager.save_checkpoint(model, epoch, val_metrics)
```

---

## Part 9: Evaluation & Metrics

**What you'll learn:** Comprehensively evaluate models using multiple metrics and generate detailed reports.

### Step 1: Compute Classification Metrics
Calculate precision, recall, F1, AUC, and confusion matrices:

- **Accuracy**: Overall correctness (can be misleading with imbalanced data)
- **Precision**: Of predicted fakes, how many are actually fake
- **Recall**: Of actual fakes, how many did we catch
- **F1**: Harmonic mean of precision and recall
- **AUC-ROC**: How well the model separates classes at all thresholds

**Action**: Create `src/deepfake_detector/utils/metrics.py`:
```python
import numpy as np
from sklearn.metrics import (
    confusion_matrix, classification_report, roc_auc_score,
    precision_recall_curve, auc, roc_curve, f1_score
)
import matplotlib.pyplot as plt

class MetricsCalculator:
    """Calculate comprehensive evaluation metrics"""
    
    @staticmethod
    def compute_metrics(y_true, y_pred, y_proba=None):
        """
        Compute classification metrics
        
        Args:
            y_true: Ground truth labels [N]
            y_pred: Predicted labels [N]
            y_proba: Predicted probabilities [N, 2] for class 1
        
        Returns:
            Dictionary of metrics
        """
        metrics = {
            'accuracy': np.mean(y_pred == y_true),
            'f1': f1_score(y_true, y_pred, average='weighted'),
            'f1_macro': f1_score(y_true, y_pred, average='macro'),
            'f1_binary': f1_score(y_true, y_pred, average='binary') if len(np.unique(y_true)) == 2 else None,
        }
        
        # AUC if probabilities available
        if y_proba is not None:
            if y_proba.ndim > 1:
                y_proba = y_proba[:, 1]  # Use probability of positive class
            metrics['auc_roc'] = roc_auc_score(y_true, y_proba)
            
            # Precision-recall AUC
            precision, recall, _ = precision_recall_curve(y_true, y_proba)
            metrics['auc_pr'] = auc(recall, precision)
        
        return metrics
    
    @staticmethod
    def print_classification_report(y_true, y_pred, class_names=None):
        """Print detailed classification report"""
        if class_names is None:
            class_names = ['FAKE', 'REAL']
        
        print("\n" + "="*60)
        print("CLASSIFICATION REPORT")
        print("="*60)
        print(classification_report(y_true, y_pred, target_names=class_names))
    
    @staticmethod
    def plot_confusion_matrix(y_true, y_pred, class_names=None, save_path=None):
        """Plot and optionally save confusion matrix"""
        if class_names is None:
            class_names = ['FAKE', 'REAL']
        
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(8, 6))
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title('Confusion Matrix')
        plt.colorbar()
        tick_marks = np.arange(len(class_names))
        plt.xticks(tick_marks, class_names)
        plt.yticks(tick_marks, class_names)
        
        # Add text annotations
        thresh = cm.max() / 2.
        for i, j in np.ndindex(cm.shape):
            plt.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
        
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved confusion matrix: {save_path}")
        
        plt.show()
    
    @staticmethod
    def plot_roc_curve(y_true, y_proba, save_path=None):
        """Plot ROC curve"""
        if y_proba.ndim > 1:
            y_proba = y_proba[:, 1]
        
        fpr, tpr, thresholds = roc_curve(y_true, y_proba)
        roc_auc = auc(fpr, tpr)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend(loc="lower right")
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved ROC curve: {save_path}")
        
        plt.show()
        
        return roc_auc

# Usage:
# cal = MetricsCalculator()
# metrics = cal.compute_metrics(y_true, y_pred, y_proba)
# cal.print_classification_report(y_true, y_pred)
# cal.plot_confusion_matrix(y_true, y_pred, save_path='confusion_matrix.png')
# cal.plot_roc_curve(y_true, y_proba, save_path='roc_curve.png')
```

### Step 2: Analyze Per-Class Performance
Understand how model performs on real vs fake samples:

- **Why**: Overall accuracy can hide class-specific problems
- **Real Examples Misclassified**: False negatives (false positives on real → security risk)
- **Fake Examples Misclassified**: False positives (real classified as fake → user frustration)

**Action**: Add per-class analysis:
```python
def analyze_per_class_performance(y_true, y_pred, y_proba, class_names=['FAKE', 'REAL']):
    """Analyze performance for each class separately"""
    
    for class_idx, class_name in enumerate(class_names):
        mask = y_true == class_idx
        class_true = y_true[mask]
        class_pred = y_pred[mask]
        
        if len(class_true) == 0:
            continue
        
        accuracy = np.mean(class_pred == class_true)
        total = len(class_true)
        correct = np.sum(class_pred == class_true)
        
        print(f"\n{class_name} ({total} samples):")
        print(f"  Accuracy: {accuracy:.4f} ({correct}/{total} correct)")
        print(f"  Error rate: {1-accuracy:.4f}")
        
        # Show misclassification examples
        errors = np.where(class_pred != class_true)[0]
        if len(errors) > 0:
            print(f"  Misclassified: {len(errors)}")
            print(f"    Predicted as: {[class_names[class_pred[e]] for e in errors[:5]]}...")
```

### Step 3: Generate Comprehensive Evaluation Report
Create a structured report with all metrics and insights:

- **Report Contents**: Dataset info, model architecture, training config, metrics, visualizations
- **Format**: Markdown or HTML for easy sharing
- **Purpose**: Document experimental results for reproducibility

**Action**: Create report generator:
```python
def generate_evaluation_report(
    experiment_name,
    model_type,
    dataset_info,
    metrics,
    train_config,
    output_dir='./reports'
):
    """Generate comprehensive evaluation report"""
    
    report_dir = Path(output_dir) / experiment_name
    report_dir.mkdir(parents=True, exist_ok=True)
    
    report_path = report_dir / 'evaluation_report.md'
    
    report_content = f"""# Evaluation Report: {experiment_name}

## Experiment Summary
- **Date**: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
- **Model**: {model_type}
- **Dataset**: {dataset_info.get('name', 'Unknown')}

## Model Configuration
- **Type**: {model_type}
- **Pretrained**: {train_config.get('pretrained', True)}
- **Image Size**: {train_config.get('image_size', 224)}

## Training Configuration
- **Batch Size**: {train_config.get('batch_size', 32)}
- **Epochs (Stage 1)**: {train_config.get('stage1_epochs', 10)}
- **Epochs (Stage 2)**: {train_config.get('stage2_epochs', 10)}
- **Learning Rate (Stage 1)**: {train_config.get('stage1_lr', 1e-4)}
- **Learning Rate (Stage 2)**: {train_config.get('stage2_lr', 1e-5)}

## Dataset Information
- **Train Samples**: {dataset_info.get('train_samples', 0)}
- **Validation Samples**: {dataset_info.get('val_samples', 0)}
- **Test Samples**: {dataset_info.get('test_samples', 0)}
- **Classes**: {', '.join(dataset_info.get('classes', ['FAKE', 'REAL']))}

## Performance Metrics

### Overall Metrics
- **Accuracy**: {metrics.get('accuracy', 0):.4f}
- **F1 Score**: {metrics.get('f1', 0):.4f}
- **AUC-ROC**: {metrics.get('auc_roc', 0):.4f}
- **AUC-PR**: {metrics.get('auc_pr', 0):.4f}

### Per-Class Metrics
- **FAKE - Precision**: {metrics.get('fake_precision', 0):.4f}
- **FAKE - Recall**: {metrics.get('fake_recall', 0):.4f}
- **REAL - Precision**: {metrics.get('real_precision', 0):.4f}
- **REAL - Recall**: {metrics.get('real_recall', 0):.4f}

## Visualizations
- Confusion Matrix: [confusion_matrix.png](confusion_matrix.png)
- ROC Curve: [roc_curve.png](roc_curve.png)
- Training Curves: [training_curves.png](training_curves.png)

## Key Insights
1. Model achieves {metrics.get('accuracy', 0):.1%} accuracy on test set
2. Strong AUC score ({metrics.get('auc_roc', 0):.3f}) indicates good separation between classes
3. Balance between precision and recall is maintained

## Recommendations
- Consider using multi-scale training for robustness
- Evaluate on videos to test temporal consistency
- Test on compressed/low-quality videos (typical in real deepfakes)
- Compare with other baselines (ResNeXt, MobileNet, etc.)

---
*Generated by Deepfake Detection Training Pipeline*
"""
    
    with open(report_path, 'w') as f:
        f.write(report_content)
    
    print(f"Report saved: {report_path}")
    return report_path

# Usage:
# generate_evaluation_report(
#     experiment_name='efficientnet_v1',
#     model_type='EfficientNet-B3',
#     dataset_info={'name': 'FaceForensics', 'train_samples': 1000},
#     metrics=metrics,
#     train_config=config
# )
```

---

## Part 10: Production Optimization

**What you'll learn:** Prepare models for real-world deployment with optimization and inference acceleration.

### Step 1: Export Models to ONNX Format
Convert PyTorch models to ONNX for cross-platform deployment:

- **ONNX**: Open Neural Network Exchange format (runs on any framework)
- **Benefits**: Inference on CPU, mobile, edge devices; framework-agnostic
- **Process**: Convert model → optimize → quantize → deploy

**Action**: Create inference script with ONNX export:
```python
import torch
import onnx

def export_to_onnx(model, output_path, image_size=224, opset_version=14):
    """Export PyTorch model to ONNX format"""
    
    model.eval()
    
    # Create dummy input
    dummy_input = torch.randn(1, 3, image_size, image_size)
    
    # Export
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        opset_version=opset_version,
        input_names=['image'],
        output_names=['logits'],
        dynamic_axes={
            'image': {0: 'batch_size'},
            'logits': {0: 'batch_size'}
        },
        verbose=False
    )
    
    # Verify ONNX model
    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)
    
    print(f"Exported to ONNX: {output_path}")
    print(f"Model graph inputs: {[input.name for input in onnx_model.graph.input]}")
    print(f"Model graph outputs: {[output.name for output in onnx_model.graph.output]}")

# Usage:
# model = EfficientNetClassifier()
# model.load_state_dict(torch.load('model.pt'))
# export_to_onnx(model, 'model.onnx')
```

### Step 2: Implement ONNX-based Inference
Create inference pipeline using ONNX Runtime for faster execution:

- **Speed**: ONNX Runtime is optimized for inference
- **Compatibility**: Run on CPU without PyTorch dependency
- **Deployment**: Easy integration into production systems

**Action**: Create `src/deepfake_detector/inference/onnx_predictor.py`:
```python
import numpy as np
import ort runtime as ort
from PIL import Image
import torch
from torchvision import transforms

class ONNXPredictor:
    """ONNX-based inference for production"""
    
    def __init__(self, model_path, class_names=None):
        # Load ONNX model
        self.session = ort.InferenceSession(model_path)
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        
        self.class_names = class_names or ['FAKE', 'REAL']
        
        # Preprocessing
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    
    def predict(self, image_path):
        """Predict on single image"""
        # Load and preprocess image
        image = Image.open(image_path).convert('RGB')
        input_tensor = self.transform(image)
        input_np = input_tensor.unsqueeze(0).numpy()
        
        # Inference
        outputs = self.session.run([self.output_name], {self.input_name: input_np})
        logits = outputs[0][0]
        
        # Softmax
        probs = torch.softmax(torch.from_numpy(logits), dim=0).numpy()
        pred_class = np.argmax(probs)
        confidence = probs[pred_class]
        
        return {
            'class': self.class_names[pred_class],
            'confidence': float(confidence),
            'probabilities': {
                self.class_names[i]: float(probs[i])
                for i in range(len(self.class_names))
            }
        }

# Usage:
# predictor = ONNXPredictor('model.onnx', class_names=['FAKE', 'REAL'])
# result = predictor.predict('test_image.jpg')
# print(f"{result['class']}: {result['confidence']:.2%}")
```

### Step 3: Implement Model Quantization
Reduce model size and increase inference speed with quantization:

- **Quantization**: Convert float32 weights to int8 (4x smaller)
- **Speed**: 2-4x faster inference on quantized models
- **Accuracy**: Minimal loss in accuracy with proper quantization

**Action**: Create quantization script:
```python
import torch
from torch.quantization import quantize_dynamic, quantize_static
import torch.nn as nn

def quantize_model_dynamic(model, output_path):
    """
    Dynamic quantization (easiest, no calibration needed)
    Good for CPU inference
    """
    quantized_model = quantize_dynamic(
        model,
        qconfig_spec={nn.Linear, nn.Conv2d},
        dtype=torch.qint8
    )
    torch.save(quantized_model.state_dict(), output_path)
    print(f"Saved quantized model: {output_path}")
    
    # Check size reduction
    original_size = sum(p.numel() for p in model.parameters())
    quantized_size = sum(p.numel() for p in quantized_model.parameters())
    print(f"Size reduction: {original_size / quantized_size:.1f}x")

def quantize_model_static(model, calibration_loader, output_path):
    """
    Static quantization (better accuracy)
    Requires calibration data
    """
    model.eval()
    model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
    
    torch.quantization.prepare_qat(model, inplace=True)
    
    # Calibration
    with torch.no_grad():
        for images, _ in calibration_loader:
            _ = model(images)
    
    torch.quantization.convert(model, inplace=True)
    
    torch.save(model.state_dict(), output_path)
    print(f"Saved statically quantized model: {output_path}")

# Benchmark before/after:
def benchmark_inference(model, device, num_runs=100):
    """Benchmark inference speed"""
    import time
    
    model.eval()
    dummy_input = torch.randn(1, 3, 224, 224).to(device)
    
    with torch.no_grad():
        # Warmup
        for _ in range(10):
            _ = model(dummy_input)
        
        # Benchmark
        start = time.time()
        for _ in range(num_runs):
            _ = model(dummy_input)
        elapsed = time.time() - start
    
    avg_time = (elapsed / num_runs) * 1000  # ms
    print(f"Average inference time: {avg_time:.2f}ms")
    return avg_time

# Usage:
# quantize_model_dynamic(model, 'model_quantized.pt')
# benchmark_inference(model, device='cpu')
# benchmark_inference(quantized_model, device='cpu')
```

---

## Summary & Next Steps

### What You've Learned

**Part 1-3**: Foundation and model understanding
- Current project structure and components
- EfficientNet vs XceptionNet architectures
- When to use each model

**Part 4-5**: Data and infrastructure
- FaceForensics dataset structure and download
- Data preprocessing pipeline (frame extraction, face detection)
- PyTorch model wrappers and configuration

**Part 6-7**: Training techniques
- Advanced augmentation (Mixup, CutMix, TTA)
- Multi-stage transfer learning strategy
- Learning rate scheduling

**Part 8-9**: Training and evaluation
- End-to-end training with monitoring
- Comprehensive metrics and reporting
- Per-class performance analysis

**Part 10**: Production
- ONNX export for deployment
- Quantization for speed and size
- Inference optimization

### Recommended Execution Order

1. **Start Simple**: Use MobileNetV2 (TensorFlow) first as baseline
2. **Try EfficientNet-B3**: Good balance of speed/accuracy
3. **Final Model**: XceptionNet for highest accuracy (if latency allows)

### Expected Performance Targets

| Model | Accuracy | AUC | Speed |
|-------|----------|-----|-------|
| MobileNetV2 | ~92% | 0.97 | Fast |
| EfficientNet-B3 | ~95% | 0.98 | Medium |
| XceptionNet | ~96% | 0.99 | Medium-Slow |

### Common Issues & Solutions

**Issue: Low validation accuracy on FaceForensics**
- Solution: More augmentation, longer training, freeze fewer layers

**Issue: Overfitting (train 99%, val 80%)**
- Solution: Increase augmentation, use dropout, reduce model size, add regularization

**Issue: Class imbalance (more real than fake)**
- Solution: Use class_weights, adjust threshold dynamically, SMOTE

**Issue: Slow training**
- Solution: Use smaller EfficientNet (B0), reduce image size to 192, mixed-precision training

### Further Reading

- EfficientNet paper: https://arxiv.org/abs/1905.11946
- Xception paper: https://arxiv.org/abs/1610.02357
- FaceForensics paper: https://arxiv.org/abs/1901.08971
- Transfer learning best practices: https://cs231n.github.io/transfer-learning/

---

**Happy training! Good luck with your deepfake detection models! 🚀**
