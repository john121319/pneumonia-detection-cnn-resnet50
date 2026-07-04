# Pneumonia Detection from Chest X-ray Images using CNN, ResNet50, Grad-CAM, and Federated Learning

This project focuses on pneumonia detection from chest X-ray images using deep learning and federated learning.

The project compares:

* A custom CNN baseline
* ResNet50 transfer learning
* ResNet50 with data augmentation
* Fine-tuned ResNet50
* Grad-CAM explainability
* A multi-client federated CNN using Federated Averaging

The long-term research direction of this project is:

**Federated Learning for Privacy-Preserving Medical Image Classification**

---

## Project Motivation

Medical image datasets are often distributed across hospitals and institutions.

A centralized deep learning workflow typically requires collecting all images in one location:

```text
Hospital A data
Hospital B data
Hospital C data
        ↓
Central server
        ↓
Model training
```

This can create privacy, governance, and data-sharing concerns.

Federated learning provides a different training strategy:

```text
Hospital A trains locally
Hospital B trains locally
Hospital C trains locally
        ↓
Model updates are aggregated
        ↓
Global model
```

In this project, the centralized deep learning experiments are first developed and evaluated. The project is then extended into a simulated three-client federated learning environment.

---

## Main Objectives

The project aims to:

1. Build a CNN baseline for pneumonia classification
2. Apply ResNet50 transfer learning
3. Use data augmentation to improve generalization
4. Fine-tune pretrained ResNet50 layers
5. Compare centralized models using medical classification metrics
6. Apply Grad-CAM for explainability
7. Simulate multiple federated clients
8. Train local client models
9. Aggregate client weights using Federated Averaging
10. Evaluate the best global federated model
11. Analyze confusion matrix and ROC-AUC performance

---

## Dataset

The dataset contains chest X-ray images belonging to two classes:

* `NORMAL`
* `PNEUMONIA`

Dataset structure:

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

| Split      | Number of Images |
| ---------- | ---------------: |
| Train      |            5,216 |
| Validation |               16 |
| Test       |              624 |

The test set contains:

| Class     | Images |
| --------- | -----: |
| NORMAL    |    234 |
| PNEUMONIA |    390 |
| Total     |    624 |

> Note: The original validation set is very small, containing only 16 images. Therefore, validation metrics may fluctuate considerably between epochs.

---

## Project Structure

```text
pneumonia-detection-cnn-resnet50/
├── chest_xray/
│   ├── train/
│   ├── val/
│   └── test/
│
├── cnn_baseline/
│   ├── model.py
│   ├── train.py
│   └── predict.py
│
├── resnet50_transfer/
│   ├── model.py
│   ├── train.py
│   ├── fine_tune.py
│   ├── predict.py
│   └── gradcam.py
│
├── evaluation/
│   └── evaluate_models.py
│
├── federated_learning/
│   ├── create_clients.py
│   ├── model.py
│   ├── train_federated.py
│   ├── evaluate_federated.py
│   └── clients_data/
│
├── results/
│   ├── cnn_pneumonia_detection_model.keras
│   ├── resnet50_pneumonia_augmented_model.keras
│   ├── resnet50_pneumonia_finetuned_model.keras
│   ├── federated_cnn_best_model.keras
│   ├── federated_cnn_final_model.keras
│   ├── federated_training_history.csv
│   ├── pneumonia_gradcam_original.png
│   ├── pneumonia_gradcam_heatmap.png
│   ├── pneumonia_gradcam_output.png
│   ├── cnn_baseline_confusion_matrix.png
│   ├── resnet50_augmented_confusion_matrix.png
│   ├── resnet50_fine_tuned_confusion_matrix.png
│   ├── federated_cnn_best_confusion_matrix.png
│   └── federated_cnn_best_roc_curve.png
│
├── test_images/
├── requirements.txt
├── .gitignore
└── README.md
```

The dataset, generated client data, and large trained model files should normally be excluded from GitHub using `.gitignore`.

---

# Part 1 — CNN Baseline

## CNN Architecture

The CNN baseline was built from scratch using TensorFlow/Keras.

Input shape:

```text
128 × 128 × 1
```

The model uses grayscale chest X-ray images.

Architecture:

```text
Input
↓
Conv2D(32)
↓
MaxPooling2D
↓
Conv2D(64)
↓
MaxPooling2D
↓
Conv2D(128)
↓
MaxPooling2D
↓
Flatten
↓
Dense
↓
Dropout
↓
Sigmoid Output
```

The output layer contains one neuron:

```text
0 → NORMAL
1 → PNEUMONIA
```

The model uses:

```text
Loss: Binary Crossentropy
Optimizer: Adam
```

---

# Part 2 — ResNet50 Transfer Learning

## Why ResNet50?

ResNet50 is a deep residual neural network pretrained on ImageNet.

Instead of training all convolutional features from scratch, pretrained ResNet50 features are reused for chest X-ray classification.

The model uses:

```text
224 × 224 × 3
```

input images.

Although the X-rays are visually grayscale, they are loaded as RGB because ImageNet-pretrained ResNet50 expects 3 channels.

---

## Transfer Learning Workflow

During the first training stage:

```text
ResNet50 backbone → frozen
Classifier head → trainable
```

Workflow:

```text
Chest X-ray
↓
Data Augmentation
↓
ResNet50 preprocessing
↓
Frozen ResNet50
↓
GlobalAveragePooling2D
↓
Dense Layer
↓
Dropout
↓
Sigmoid Output
```

---

# Part 3 — Data Augmentation

The ResNet50 model includes augmentation such as:

* Random horizontal flip
* Random rotation
* Random zoom

Example:

```text
Original X-ray
     ↓
Random transformation
     ↓
New training variation
```

The goal is to reduce overfitting and improve generalization.

---

# Part 4 — Fine-Tuning

After initial transfer learning, the trained model is loaded again.

The last ResNet50 layers are unfrozen and trained using a smaller learning rate:

```text
Learning rate = 1e-5
```

Fine-tuning workflow:

```text
Train classifier with frozen backbone
                ↓
Load trained model
                ↓
Unfreeze selected ResNet50 layers
                ↓
Recompile with small learning rate
                ↓
Continue training
```

This allows the pretrained feature extractor to adapt more closely to chest X-ray images.

---

# Part 5 — Centralized Model Evaluation

The models were evaluated on the same 624-image test set.

## Overall Results

| Model                      | Accuracy | Pneumonia Precision | Pneumonia Recall | Pneumonia F1-score |
| -------------------------- | -------: | ------------------: | ---------------: | -----------------: |
| CNN Baseline               |   75.48% |                 72% |              99% |                84% |
| ResNet50 with Augmentation |   86.06% |                 84% |              97% |                90% |
| ResNet50 Fine-Tuned        |   86.22% |                 83% |              97% |                90% |

---

## CNN Baseline Classification Report

```text
              precision    recall  f1-score   support

      NORMAL       0.98      0.35      0.52       234
   PNEUMONIA       0.72      0.99      0.84       390

    accuracy                           0.75       624
   macro avg       0.85      0.67      0.68       624
weighted avg       0.82      0.75      0.72       624
```

### Interpretation

The CNN baseline achieved very high pneumonia recall but low NORMAL recall.

This indicates that the model was highly sensitive to pneumonia but produced many false-positive pneumonia predictions.

---

## ResNet50 with Augmentation Classification Report

```text
              precision    recall  f1-score   support

      NORMAL       0.92      0.68      0.79       234
   PNEUMONIA       0.84      0.97      0.90       390

    accuracy                           0.86       624
   macro avg       0.88      0.83      0.84       624
weighted avg       0.87      0.86      0.86       624
```

---

## Fine-Tuned ResNet50 Classification Report

```text
              precision    recall  f1-score   support

      NORMAL       0.94      0.68      0.79       234
   PNEUMONIA       0.83      0.97      0.90       390

    accuracy                           0.86       624
   macro avg       0.89      0.82      0.84       624
weighted avg       0.87      0.86      0.86       624
```

---

## Centralized Experiment Interpretation

The CNN baseline achieved:

```text
Accuracy: 75.48%
Pneumonia Recall: 99%
```

However, its low NORMAL recall indicates a large number of false positives.

The ResNet50 models improved overall performance substantially:

```text
Augmented ResNet50 Accuracy: 86.06%
Fine-Tuned ResNet50 Accuracy: 86.22%
```

Both ResNet50 variants achieved approximately:

```text
Pneumonia Recall: 97%
Pneumonia F1-score: 90%
```

The fine-tuned ResNet50 achieved the highest centralized test accuracy.

---

# Part 6 — Grad-CAM Explainability

Grad-CAM was applied to the fine-tuned ResNet50 model.

Grad-CAM generates a heatmap showing which image regions contributed most strongly to a prediction.

Workflow:

```text
Chest X-ray
↓
ResNet50 prediction
↓
Gradient computation
↓
Convolutional activation weighting
↓
Heatmap
↓
Overlay
```

Generated outputs:

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

Grad-CAM provides model interpretability, but activation maps should be interpreted cautiously. A technically valid heatmap does not automatically prove clinically meaningful localization.

---

# Part 7 — Federated Learning

## Federated Learning Objective

The centralized models use a single training dataset.

The federated experiment instead simulates multiple hospitals:

```text
Client 1
Client 2
Client 3
```

Each client trains locally on its own data partition.

The raw local training images are not pooled into a central training process.

---

## Federated Workflow

```text
Global Model
    ↓
Send global weights to clients
    ↓
Client 1 trains locally
Client 2 trains locally
Client 3 trains locally
    ↓
Collect client model weights
    ↓
Federated Averaging
    ↓
Update global model
    ↓
Evaluate global model
```

This process is repeated over multiple communication rounds.

---

## Simulated Federated Clients

The original training set was divided among 3 simulated clients.

Approximate client sizes:

| Client   | Training Images |
| -------- | --------------: |
| Client 1 |           1,738 |
| Client 2 |           1,738 |
| Client 3 |           1,740 |

Total:

```text
5,216 training images
```

The current experiment represents an approximately IID-like client partition because the clients have similar sample sizes and class distributions.

---

## Federated CNN

A lightweight CNN was used for the federated experiment.

Input shape:

```text
128 × 128 × 1
```

The model uses:

* Conv2D
* MaxPooling2D
* Dense layers
* Dropout
* Binary sigmoid output

A smaller CNN was selected instead of ResNet50 to reduce local training cost across multiple simulated clients.

---

## Weighted Federated Averaging

The improved federated implementation uses weighted averaging.

Instead of giving every client exactly equal influence, the aggregation considers client sample count:

```text
Client contribution
        ∝
Number of local training samples
```

Conceptually:

```text
New Global Weights
=
Weighted average of client weights
```

Because the three clients currently have very similar dataset sizes, their aggregation contributions are also very similar.

---

# Part 8 — Federated Training Results

The federated model was trained for 5 communication rounds.

Latest experiment:

| Round       |       Loss |   Accuracy |  Precision |     Recall |
| ----------- | ---------: | ---------: | ---------: | ---------: |
| Initial     |     0.6900 |     62.50% |     62.50% |    100.00% |
| **Round 1** | **0.3709** | **84.13%** | **81.16%** | **97.18%** |
| Round 2     |     0.5018 |     80.29% |     77.08% |     97.44% |
| Round 3     |     0.5177 |     80.61% |     77.28% |     97.69% |
| Round 4     |     0.6004 |     80.61% |     77.06% |     98.21% |
| Round 5     |     0.4497 |     83.81% |     80.81% |     97.18% |

Best round:

```text
Round 1
```

Best accuracy:

```text
84.13%
```

The best global model was automatically saved as:

```text
results/federated_cnn_best_model.keras
```

The final round model was separately saved as:

```text
results/federated_cnn_final_model.keras
```

---

# Part 9 — Best Federated Model Evaluation

The best federated model achieved:

```text
Accuracy: 84.13%
Precision: 81.16%
Pneumonia Recall: 97.18%
ROC-AUC: 0.9442
```

Classification report:

```text
              precision    recall  f1-score   support

      NORMAL     0.9299    0.6239    0.7468       234
   PNEUMONIA     0.8116    0.9718    0.8845       390

    accuracy                         0.8413       624
   macro avg     0.8707    0.7979    0.8156       624
weighted avg     0.8560    0.8413    0.8329       624
```

---

## Federated Confusion Matrix

```text
                 Predicted
              NORMAL  PNEUMONIA

Actual NORMAL    146      88
Actual PNEUMONIA  11     379
```

Interpretation:

### True Negatives

```text
146
```

146 NORMAL X-rays were correctly classified as NORMAL.

### False Positives

```text
88
```

88 NORMAL X-rays were incorrectly classified as PNEUMONIA.

### False Negatives

```text
11
```

11 PNEUMONIA X-rays were incorrectly classified as NORMAL.

### True Positives

```text
379
```

379 PNEUMONIA X-rays were correctly classified as PNEUMONIA.

---

## Sensitivity and Specificity

Pneumonia sensitivity:

```text
97.18%
```

This means the model detected 379 of 390 pneumonia cases.

Specificity:

```text
62.39%
```

This means the model correctly identified 146 of 234 normal cases.

The model therefore demonstrates:

```text
High pneumonia sensitivity
Lower specificity
```

This indicates a screening-oriented behavior with relatively few missed pneumonia cases but a higher number of false-positive alerts.

---

## ROC-AUC

The best federated CNN achieved:

```text
ROC-AUC = 0.9442
```

ROC-AUC evaluates discrimination performance across many possible decision thresholds.

This result indicates strong class-separation ability, even though the default threshold of `0.5` still produces a substantial number of false-positive predictions.

---

# Part 10 — Centralized vs Federated Comparison

| Model               | Accuracy | Pneumonia Precision | Pneumonia Recall | Pneumonia F1 |
| ------------------- | -------: | ------------------: | ---------------: | -----------: |
| CNN Baseline        |   75.48% |                 72% |              99% |          84% |
| ResNet50 Augmented  |   86.06% |                 84% |              97% |          90% |
| ResNet50 Fine-Tuned |   86.22% |                 83% |              97% |          90% |
| Federated CNN       |   84.13% |              81.16% |           97.18% |       88.45% |

The fine-tuned ResNet50 achieved the highest centralized accuracy.

The federated CNN achieved:

```text
84.13% accuracy
97.18% pneumonia recall
0.9442 ROC-AUC
```

while simulating decentralized local training across three clients.

The results do not demonstrate that federated learning is universally superior to centralized learning. Instead, they show that competitive classification performance can be maintained in a simulated multi-client training environment.

---

# Research Observations

## 1. Accuracy Alone Is Misleading

The untrained initial global model achieved approximately:

```text
62.5% accuracy
```

This occurred because the test set contains:

```text
390 PNEUMONIA images out of 624 total images
```

which is exactly:

```text
62.5%
```

The initial model strongly favored the majority class.

This demonstrates why accuracy alone can be misleading on imbalanced medical datasets.

---

## 2. More Federated Rounds Did Not Always Improve Performance

The best global test accuracy occurred before the final round.

This demonstrates that:

```text
More communication rounds
≠
Guaranteed better global generalization
```

Possible explanations include:

* Local overfitting
* Client drift
* Stochastic optimization
* Aggregation instability
* Global generalization mismatch

---

## 3. Local Accuracy Can Increase While Global Accuracy Decreases

Local client models achieved increasingly high training accuracy over later rounds.

However, the global test model did not improve monotonically.

This suggests that stronger local optimization does not automatically produce a better aggregated global model.

---

## 4. Best-Model Checkpointing Is Important

The project saves:

```text
Best global model
```

separately from:

```text
Final round model
```

This prevents the final communication round from automatically replacing a better earlier global model.

---

## 5. Federated Learning Does Not Automatically Guarantee Complete Privacy

This project simulates decentralized training because client training images remain in local client partitions.

However, federated learning alone should not be described as complete privacy protection.

Model updates can still introduce privacy risks.

Future privacy-oriented extensions may include:

* Differential Privacy
* Secure Aggregation
* Encrypted model updates
* Membership inference evaluation

---

# How to Run the Project

## 1. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 2. Train CNN Baseline

```bash
cd cnn_baseline
python train.py
```

---

## 3. Predict with CNN Baseline

```bash
cd cnn_baseline
python predict.py
```

---

## 4. Train ResNet50 with Augmentation

```bash
cd resnet50_transfer
python train.py
```

---

## 5. Fine-Tune ResNet50

```bash
cd resnet50_transfer
python fine_tune.py
```

---

## 6. Predict with ResNet50

```bash
cd resnet50_transfer
python predict.py
```

---

## 7. Evaluate Centralized Models

```bash
cd evaluation
python evaluate_models.py
```

---

## 8. Generate Grad-CAM

```bash
cd resnet50_transfer
python gradcam.py
```

---

## 9. Create Federated Clients

```bash
cd federated_learning
python create_clients.py
```

This creates three simulated client datasets.

---

## 10. Train Federated Model

```bash
cd federated_learning
python train_federated.py
```

The script:

1. Builds a global model
2. Sends global weights to local clients
3. Trains local client models
4. Collects local weights
5. Applies weighted Federated Averaging
6. Updates the global model
7. Evaluates after each round
8. Saves the best global model
9. Saves round history to CSV

---

## 11. Evaluate Best Federated Model

```bash
cd federated_learning
python evaluate_federated.py
```

This generates:

* Classification report
* Confusion matrix
* ROC-AUC
* ROC curve

---

# Requirements

Main Python libraries:

```text
tensorflow
numpy
matplotlib
pillow
scikit-learn
```

Example installation:

```bash
pip install tensorflow numpy matplotlib pillow scikit-learn
```

---

# Technologies Used

* Python
* TensorFlow
* Keras
* NumPy
* Matplotlib
* Scikit-learn
* CNN
* ResNet50
* Transfer Learning
* Fine-Tuning
* Data Augmentation
* Grad-CAM
* Federated Averaging
* ROC-AUC Analysis

---

# Key Learning Outcomes

Through this project, I learned:

* CNN architecture design
* Binary medical image classification
* Training and test-set evaluation
* Image preprocessing
* Grayscale and RGB model input requirements
* ResNet50 transfer learning
* Data augmentation
* Fine-tuning pretrained networks
* Precision, recall, and F1-score analysis
* Confusion matrix interpretation
* Grad-CAM explainability
* Federated learning simulation
* Local client training
* Global model aggregation
* Weighted FedAvg
* Communication rounds
* Best-round checkpointing
* Class imbalance interpretation
* ROC-AUC evaluation
* Sensitivity and specificity
* Differences between local and global generalization

---

# Limitations

This project has several important limitations:

1. The validation set contains only 16 images
2. The client partition is approximately IID-like
3. The simulated clients are not real hospitals
4. No differential privacy mechanism is currently implemented
5. No secure aggregation mechanism is currently implemented
6. The federated experiment uses a lightweight CNN rather than federated ResNet50
7. The experiment has not yet been repeated across multiple random seeds
8. Threshold optimization has not yet been completed
9. Grad-CAM localization is not equivalent to clinical explanation
10. The model is a research and educational prototype, not a clinical diagnostic system

---

# Future Work

Planned future improvements include:

* Create a better validation split
* Run experiments across multiple random seeds
* Report mean performance and standard deviation
* Add threshold optimization
* Compare sensitivity and specificity across thresholds
* Simulate Non-IID client distributions
* Compare IID and Non-IID federated learning
* Compare centralized CNN and federated CNN directly
* Add class weighting
* Add client sampling
* Add partial client participation
* Test FedProx
* Test federated ResNet50
* Test EfficientNet or DenseNet
* Add Differential Privacy
* Explore Secure Aggregation
* Improve Grad-CAM localization
* Add calibration analysis
* Add precision-recall curves
* Evaluate external datasets if available

---

# Planned Research Extension

The next major experiment is:

```text
IID Federated Learning
        ↓
Non-IID Federated Learning
        ↓
IID vs Non-IID Comparison
        ↓
Centralized vs Federated Comparison
        ↓
Differential Privacy
```

This progression supports the long-term research topic:

**Federated Learning for Privacy-Preserving Medical Image Classification**

---

# Disclaimer

This project is for research, educational, and portfolio purposes only.

It is not intended for clinical diagnosis, medical decision-making, or real-world patient care.

A real clinical system would require substantially stronger validation, external testing, calibration, bias analysis, regulatory review, and expert medical evaluation.

---

# Author

**Yohannes Alelign Biresaw**

Bachelor's Degree in Electrical and Computer Engineering

Research interests:

* Artificial Intelligence
* Deep Learning
* Medical Image Analysis
* Federated Learning
* Privacy-Preserving Machine Learning
* Explainable AI
