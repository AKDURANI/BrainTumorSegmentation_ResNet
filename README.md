# Brain Tumor Classification using ResNet-50

A deep learning project for classifying brain tumor types from MRI images using transfer learning with ResNet-50 architecture.

## 🧠 Overview

This project implements a convolutional neural network to classify brain MRI images into four categories:
- **Glioma** - A type of tumor that occurs in the brain and spinal cord
- **Meningioma** - A tumor that arises from the meninges
- **No Tumor** - Normal brain scans without tumors
- **Pituitary** - Tumors that develop in the pituitary gland

## 🎯 Model Architecture

The model uses **ResNet-50** as the base architecture with transfer learning:
- Pre-trained ResNet-50 (ImageNet weights) as feature extractor
- Global Average Pooling layer
- Dropout layer (0.4) for regularization
- Dense output layer with softmax activation for 4-class classification

## 📁 Dataset Structure

```
brain_tumour/
├── Training/
│   ├── glioma/
│   ├── meningioma/
│   ├── notumor/
│   └── pituitary/
└── Testing/
    ├── glioma/
    ├── meningioma/
    ├── notumor/
    └── pituitary/
```

## 🔧 Prerequisites

### Required Libraries
```python
tensorflow
keras
scikit-learn
opencv-python
matplotlib
numpy
PIL
tqdm
imutils
```

### Installation
```bash
pip install tensorflow scikit-learn opencv-python matplotlib numpy pillow tqdm imutils
```

## 🚀 Usage

### 1. Data Preprocessing
The project includes several preprocessing steps:

**Image Cropping Function:**
```python
def crop_img(img):
    # Finds extreme points and crops the brain region
    # Removes background noise and focuses on brain tissue
```

**Image Processing Pipeline:**
- Convert to grayscale
- Apply bilateral filtering for noise reduction
- Apply colormap (BONE) for better visualization
- Resize to 200x200 pixels
- Normalize pixel values to [0,1] range

### 2. Data Augmentation
```python
demo_datagen = ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.05,
    height_shift_range=0.05,
    rescale=1./255,
    zoom_range=0.2,
    shear_range=0.05,
    brightness_range=[0.1, 1.5],
    horizontal_flip=True,
    vertical_flip=True
)
```

### 3. Model Training
```python
# Load pre-trained ResNet-50
conv_base = ResNet50(
    include_top=False,
    input_shape=(200, 200, 3),
    weights='imagenet'
)

# Compile model
adam = Adam(learning_rate=0.0001)
model.compile(
    optimizer=adam, 
    loss='categorical_crossentropy', 
    metrics=['accuracy']
)

# Train model
history = model.fit(
    datagen.flow(x_train, y_train, batch_size=64),
    validation_data=(x_val, y_val),
    epochs=50,
    callbacks=callbacks
)
```

## 📊 Model Performance

### Key Features:
- **Input Size:** 200x200x3 RGB images
- **Batch Size:** 64
- **Learning Rate:** 0.0001 (Adam optimizer)
- **Epochs:** 50
- **Validation Split:** Included for model evaluation

### Callbacks Used:
- `ModelCheckpoint` - Save best model weights
- `ReduceLROnPlateau` - Reduce learning rate when validation loss plateaus
- `EarlyStopping` - Stop training when no improvement

## 🔍 Image Processing Pipeline

1. **Load MRI images** from dataset directories
2. **Crop brain region** using contour detection
3. **Apply bilateral filtering** to reduce noise
4. **Convert to pseudocolor** using COLORMAP_BONE
5. **Resize** to standard 200x200 dimensions
6. **Normalize** pixel values
7. **Apply data augmentation** during training

## 📈 Evaluation

The model performance is evaluated using:
- **Accuracy Score**
- **Classification Report** (precision, recall, F1-score)
- **Confusion Matrix**
- **Training/Validation Loss and Accuracy Plots**

## 🛠️ File Structure

```
project/
├── main.py                 # Main training script
├── preprocessing.py        # Image preprocessing functions
├── model.py               # Model architecture
├── utils.py               # Utility functions
├── requirements.txt       # Dependencies
└── README.md             # This file
```
