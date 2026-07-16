from pathlib import Path

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_auc_score,
    roc_curve
)


FEDERATED_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = FEDERATED_DIR.parent

TEST_PATH = PROJECT_ROOT / "chest_xray" / "test"

RESULTS_PATH = PROJECT_ROOT / "results"

BEST_MODEL_PATH = (
    RESULTS_PATH
    / "federated_cnn_non_iid_best_model.keras"
)

CONFUSION_MATRIX_PATH = (
    RESULTS_PATH
    / "federated_cnn_non_iid_confusion_matrix.png"
)

ROC_CURVE_PATH = (
    RESULTS_PATH
    / "federated_cnn_non_iid_roc_curve.png"
)


IMG_SIZE = (128, 128)
BATCH_SIZE = 32
THRESHOLD = 0.5


RESULTS_PATH.mkdir(
    parents=True,
    exist_ok=True
)


def load_test_dataset():

    test_ds = tf.keras.utils.image_dataset_from_directory(
        TEST_PATH,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        color_mode="grayscale",
        label_mode="binary",
        shuffle=False
    )

    class_names = test_ds.class_names

    test_ds = test_ds.map(
        lambda x, y: (
            tf.cast(x, tf.float32) / 255.0,
            y
        ),
        num_parallel_calls=tf.data.AUTOTUNE
    )

    test_ds = test_ds.prefetch(tf.data.AUTOTUNE)

    return test_ds, class_names



def get_true_labels(dataset):

    labels = []

    for _, batch_labels in dataset:

        labels.extend(
            batch_labels.numpy().reshape(-1)
        )

    return np.array(labels, dtype=int)


def main():

    if not BEST_MODEL_PATH.exists():
        raise FileNotFoundError(
            "Non-IID best model not found. "
            "Run train_non_iid_federated.py first."
        )

    test_ds, class_names = load_test_dataset()

    print("Class names:", class_names)

    model = tf.keras.models.load_model(
        BEST_MODEL_PATH,
        compile=False
    )

    y_prob = model.predict(
        test_ds,
        verbose=1
    ).reshape(-1)

    y_true = get_true_labels(test_ds)

    y_pred = (y_prob >= THRESHOLD).astype(int)

    accuracy = accuracy_score(y_true, y_pred)

    precision = precision_score(
        y_true,
        y_pred,
        zero_division=0
    )

    recall = recall_score(
        y_true,
        y_pred,
        zero_division=0
    )

    f1 = f1_score(
        y_true,
        y_pred,
        zero_division=0
    )

    auc_score = roc_auc_score(
        y_true,
        y_prob
    )

    print("\n====================================")
    print("Non-IID Federated CNN Evaluation")
    print("====================================")

    print("Threshold:", THRESHOLD)
    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1-score:", f1)
    print("ROC-AUC:", auc_score)

    print("\nClassification Report:")

    print(
        classification_report(
            y_true,
            y_pred,
            target_names=class_names,
            digits=4,
            zero_division=0
        )
    )

    cm = confusion_matrix(
        y_true,
        y_pred
    )

    print("Confusion Matrix:")
    print(cm)

    tn, fp, fn, tp = cm.ravel()

    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    print("\nMedical Metrics")
    print("------------------------------")
    print("True Negatives:", tn)
    print("False Positives:", fp)
    print("False Negatives:", fn)
    print("True Positives:", tp)
    print("Sensitivity:", sensitivity)
    print("Specificity:", specificity)

    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=class_names
    )

    display.plot(values_format="d")

    plt.title("Non-IID Federated CNN Confusion Matrix")

    plt.tight_layout()

    plt.savefig(
        CONFUSION_MATRIX_PATH,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(
        "Confusion matrix saved to:",
        CONFUSION_MATRIX_PATH
    )

    fpr, tpr, _ = roc_curve(
        y_true,
        y_prob
    )

    plt.figure(figsize=(8, 6))

    plt.plot(
        fpr,
        tpr,
        label=f"Non-IID Federated CNN AUC = {auc_score:.4f}"
    )

    plt.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        label="Random classifier"
    )

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Non-IID Federated CNN ROC Curve")
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        ROC_CURVE_PATH,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(
        "ROC curve saved to:",
        ROC_CURVE_PATH
    )


if __name__ == "__main__":
    main()