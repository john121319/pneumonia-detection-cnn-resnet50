import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_auc_score,
    roc_curve
)



TEST_PATH = "../chest_xray/test"

BEST_MODEL_PATH = "../results/federated_cnn_best_model.keras"

RESULTS_PATH = "../results"

CONFUSION_MATRIX_PATH = (
    "../results/federated_cnn_best_confusion_matrix.png"
)

ROC_CURVE_PATH = (
    "../results/federated_cnn_best_roc_curve.png"
)



IMG_SIZE = (128, 128)
BATCH_SIZE = 32
THRESHOLD = 0.5



os.makedirs(
    RESULTS_PATH,
    exist_ok=True
)



test_ds = tf.keras.utils.image_dataset_from_directory(
    TEST_PATH,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    color_mode="grayscale",
    label_mode="binary",
    shuffle=False
)

class_names = test_ds.class_names

print(
    "Class names:",
    class_names
)



test_ds = test_ds.map(
    lambda x, y: (
        tf.cast(x, tf.float32) / 255.0,
        y
    ),
    num_parallel_calls=tf.data.AUTOTUNE
)

test_ds = test_ds.prefetch(
    tf.data.AUTOTUNE
)



model = tf.keras.models.load_model(
    BEST_MODEL_PATH
)



print("\n====================================")
print("Best Federated CNN Evaluation")
print("====================================")

results = model.evaluate(
    test_ds,
    verbose=1
)

print("\nRaw evaluation results:")
print(results)


y_true = []

for images, labels in test_ds:

    y_true.extend(
        labels.numpy().reshape(-1)
    )

y_true = np.array(
    y_true
).astype(int)


y_prob = model.predict(
    test_ds,
    verbose=1
).reshape(-1)


y_pred = (
    y_prob >= THRESHOLD
).astype(int)


print("\n====================================")
print("Classification Report")
print("====================================")

print(
    classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        digits=4
    )
)


cm = confusion_matrix(
    y_true,
    y_pred
)

print("\nConfusion Matrix:")

print(cm)


display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=class_names
)

display.plot(
    values_format="d"
)

plt.title(
    "Best Federated CNN Confusion Matrix"
)

plt.tight_layout()

plt.savefig(
    CONFUSION_MATRIX_PATH
)

plt.close()

print(
    "\nConfusion matrix saved to:",
    CONFUSION_MATRIX_PATH
)


auc_score = roc_auc_score(
    y_true,
    y_prob
)

print(
    "\nROC-AUC:",
    auc_score
)


fpr, tpr, thresholds = roc_curve(
    y_true,
    y_prob
)

plt.figure(
    figsize=(8, 6)
)

plt.plot(
    fpr,
    tpr,
    label=f"Federated CNN AUC = {auc_score:.4f}"
)

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--"
)

plt.xlabel(
    "False Positive Rate"
)

plt.ylabel(
    "True Positive Rate"
)

plt.title(
    "Best Federated CNN ROC Curve"
)

plt.legend()

plt.tight_layout()

plt.savefig(
    ROC_CURVE_PATH
)

plt.close()

print(
    "ROC curve saved to:",
    ROC_CURVE_PATH
)


print("\n====================================")
print("Evaluation Completed")
print("====================================")

print(
    "Model:",
    BEST_MODEL_PATH
)

print(
    "Threshold:",
    THRESHOLD
)

print(
    "ROC-AUC:",
    auc_score
)