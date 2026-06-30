# Pneumonia Detection from Chest X-ray Images using CNN and ResNet50

This project focuses on detecting pneumonia from chest X-ray images using deep learning.

The goal is to compare a custom CNN baseline model with ResNet50 transfer learning, data augmentation, fine-tuning, model evaluation, and Grad-CAM explainability.

This project is part of my machine learning and medical AI portfolio, with a long-term research direction toward privacy-preserving medical image classification and federated learning.

---

## Project Overview

Pneumonia detection is an important medical image classification task. Chest X-ray images are classified into two categories:

- NORMAL
- PNEUMONIA

In this project, I trained and evaluated three deep learning models:

1. CNN Baseline
2. ResNet50 with Data Augmentation
3. Fine-Tuned ResNet50

I also used Grad-CAM to visualize which image regions influenced the ResNet50 prediction.

---

## Dataset Structure

The dataset is organized as follows:

```text
chest_xray/
├── train/
│   ├── NORMAL/
│   └── PNEUMONIA/
├── val/
│   ├── NORMAL/
│   └── PNEUMONIA/
└── test/
    ├── NORMAL/
    └── PNEUMONIA/
```

Dataset summary:

| Split | Images |
|---|---:|
| Train | 5,216 |
| Validation | 16 |
| Test | 624 |

Note: The validation set is very small, so validation accuracy may fluctuate during training.

---

## Project Structure

```text
pneumonia-detection-cnn-resnet50/
├── chest_xray/                  # Dataset folder, not uploaded to GitHub
├── cnn_baseline/
│   ├── model.py
│   ├── train.py
│   └── predict.py
├── resnet50_transfer/
│   ├── model.py
│   ├── train.py
│   ├── fine_tune.py
│   ├── predict.py
│   └── gradcam.py
├── evaluation/
│   └── evaluate_models.py
├── results/                     # Model outputs and result images
├── test_images/
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Models Used

### 1. CNN Baseline

The CNN baseline was built from scratch using TensorFlow/Keras.

Input format:

```text
128 × 128 × 1
```

The CNN used grayscale chest X-ray images.

Main layers:

- Conv2D
- MaxPooling2D
- Flatten
- Dense
- Dropout
- Sigmoid output layer

---

### 2. ResNet50 with Data Augmentation

ResNet50 was used with ImageNet pretrained weights.

Input format:

```text
224 × 224 × 3
```

Although chest X-ray images are visually grayscale, they were loaded as RGB because ImageNet-pretrained ResNet50 requires 3 input channels.

Data augmentation included:

- Random horizontal flip
- Random rotation
- Random zoom

During the first ResNet50 training stage, the ResNet50 backbone was frozen and only the custom classifier head was trained.

---

### 3. Fine-Tuned ResNet50

After transfer learning, the trained ResNet50 model was loaded again and the last ResNet50 layers were unfrozen.

Fine-tuning was done using a small learning rate:

```text
1e-5
```

This allows the model to carefully adapt pretrained ResNet50 features to chest X-ray images without destroying the pretrained knowledge.

---

## Grad-CAM Explainability

Grad-CAM was used to visualize which regions of the chest X-ray image influenced the model prediction.

Grad-CAM helps check whether the model focuses on relevant chest/lung regions instead of irrelevant image borders or background areas.

Generated Grad-CAM outputs:

```text
results/pneumonia_gradcam_original.png
results/pneumonia_gradcam_heatmap.png
results/pneumonia_gradcam_output.png
```

Example prediction:

```text
Pneumonia probability: 0.99997467
Prediction: PNEUMONIA
```

---

## Results

The models were evaluated on the test set containing 624 chest X-ray images.

| Model | Accuracy | Pneumonia Precision | Pneumonia Recall | Pneumonia F1-score |
|---|---:|---:|---:|---:|
| CNN Baseline | 75.48% | 72% | 99% | 84% |
| ResNet50 with Augmentation | 86.06% | 84% | 97% | 90% |
| ResNet50 Fine-Tuned | 86.22% | 83% | 97% | 90% |

---

## Result Interpretation

The CNN baseline achieved high pneumonia recall, but it performed poorly on NORMAL images. This means the CNN predicted many normal images as pneumonia.

The ResNet50 models performed better overall. They improved accuracy, precision, and F1-score compared to the CNN baseline.

The fine-tuned ResNet50 model achieved the highest test accuracy:

```text
86.22%
```

The ResNet50 model with augmentation achieved a lower test loss:

```text
0.3514
```

For medical screening tasks, pneumonia recall is especially important because missing pneumonia cases can be more harmful than false alarms.

Both ResNet50 models achieved strong pneumonia recall:

```text
97%
```

---

## Classification Reports

### CNN Baseline

```text
              precision    recall  f1-score   support

      NORMAL       0.98      0.35      0.52       234
   PNEUMONIA       0.72      0.99      0.84       390

    accuracy                           0.75       624
   macro avg       0.85      0.67      0.68       624
weighted avg       0.82      0.75      0.72       624
```

### ResNet50 with Augmentation

```text
              precision    recall  f1-score   support

      NORMAL       0.92      0.68      0.79       234
   PNEUMONIA       0.84      0.97      0.90       390

    accuracy                           0.86       624
   macro avg       0.88      0.83      0.84       624
weighted avg       0.87      0.86      0.86       624
```

### ResNet50 Fine-Tuned

```text
              precision    recall  f1-score   support

      NORMAL       0.94      0.68      0.79       234
   PNEUMONIA       0.83      0.97      0.90       390

    accuracy                           0.86       624
   macro avg       0.89      0.82      0.84       624
weighted avg       0.87      0.86      0.86       624
```

---

## How to Run the Project

### 1. Install Requirements

```bash
pip install -r requirements.txt
```

### 2. Train CNN Baseline

```bash
cd cnn_baseline
python train.py
```

### 3. Train ResNet50 with Augmentation

```bash
cd resnet50_transfer
python train.py
```

### 4. Fine-Tune ResNet50

```bash
cd resnet50_transfer
python fine_tune.py
```

### 5. Make Prediction

```bash
cd resnet50_transfer
python predict.py
```

### 6. Evaluate All Models

```bash
cd evaluation
python evaluate_models.py
```

### 7. Generate Grad-CAM

```bash
cd resnet50_transfer
python gradcam.py
```

---

## Key Learning Outcomes

Through this project, I learned:

- How to build a CNN baseline for medical image classification
- How to use ResNet50 transfer learning
- Why pretrained ResNet50 requires RGB input
- How to use data augmentation
- How to fine-tune pretrained models
- How to evaluate models using precision, recall, F1-score, and confusion matrix
- Why recall is important in medical screening
- How to use Grad-CAM for explainable AI

---

## Technologies Used

- Python
- TensorFlow
- Keras
- NumPy
- Matplotlib
- Scikit-learn
- ResNet50
- Grad-CAM

---

## Future Improvements

Possible future improvements include:

- Creating a better validation split from the training data
- Applying class weights to handle class imbalance
- Testing other pretrained models such as EfficientNet or DenseNet
- Improving Grad-CAM localization
- Adding ROC curve and AUC score
- Building a federated learning version for privacy-preserving medical image classification

---

## Research Direction

This project is a foundation for my future research interest:

```text
Federated Learning for Privacy-Preserving Medical Image Classification
```

The next step is to extend this project into a federated learning setup where multiple clients train a shared model without directly sharing medical image data.