import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay


TEST_PATH = "../chest_xray/test"

CNN_MODEL_PATH = "../results/cnn_pneumonia_detection_model.keras"
RESNET_AUG_MODEL_PATH = "../results/resnet50_pneumonia_augmented_model.keras"
RESNET_FINE_TUNED_MODEL_PATH = "../results/resnet50_pneumonia_finetuned_model.keras"

RESULTS_PATH = "../results"



BATCH_SIZE = 32

CNN_IMAGE_SIZE = (128, 128)
RESNET_IMAGE_SIZE = (224, 224)



os.makedirs(RESULTS_PATH, exist_ok=True)



def load_cnn_test_data():

    test_ds = tf.keras.utils.image_dataset_from_directory(
        TEST_PATH,
        image_size=CNN_IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        color_mode="grayscale",
        label_mode="binary",
        shuffle=False
    )

    class_names = test_ds.class_names

    test_ds = test_ds.map(
        lambda x, y: (x / 255.0, y)
    )

    return test_ds, class_names



def load_resnet_test_data():

    test_ds = tf.keras.utils.image_dataset_from_directory(
        TEST_PATH,
        image_size=RESNET_IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        color_mode="rgb",
        label_mode="binary",
        shuffle=False
    )

    class_names = test_ds.class_names

    return test_ds, class_names



def get_true_labels(dataset):

    labels = []

    for images, batch_labels in dataset:
        labels.extend(batch_labels.numpy())

    labels = np.array(labels).reshape(-1)

    return labels.astype(int)



def get_predictions(model, dataset):

    probabilities = model.predict(dataset)

    probabilities = probabilities.reshape(-1)

    predictions = (probabilities > 0.5).astype(int)

    return predictions



def evaluate_model(model_path, dataset, class_names, model_name):

    if not os.path.exists(model_path):
        print("\nSkipping:", model_name)
        print("Model file not found:", model_path)
        return

    print("\n====================================")
    print(model_name)
    print("====================================")

    model = tf.keras.models.load_model(model_path)

    results = model.evaluate(dataset)

    print("Metric names:", model.metrics_names)
    print("Evaluation results:", results)

    for name, value in zip(model.metrics_names, results):
        print(f"{name}: {value}")

    y_true = get_true_labels(dataset)

    y_pred = get_predictions(model, dataset)

    print("\nClassification Report:")
    print(
        classification_report(
            y_true,
            y_pred,
            target_names=class_names
        )
    )

    cm = confusion_matrix(y_true, y_pred)

    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=class_names
    )

    display.plot(values_format="d")

    plt.title(f"{model_name} Confusion Matrix")

    save_name = model_name.lower().replace(" ", "_").replace("-", "_")

    save_path = os.path.join(
        RESULTS_PATH,
        f"{save_name}_confusion_matrix.png"
    )

    plt.savefig(save_path)

    plt.close()

    print("Confusion matrix saved to:", save_path)


cnn_test_ds, class_names = load_cnn_test_data()

resnet_test_ds, _ = load_resnet_test_data()


evaluate_model(
    CNN_MODEL_PATH,
    cnn_test_ds,
    class_names,
    "CNN Baseline"
)


evaluate_model(
    RESNET_AUG_MODEL_PATH,
    resnet_test_ds,
    class_names,
    "ResNet50 Augmented"
)


evaluate_model(
    RESNET_FINE_TUNED_MODEL_PATH,
    resnet_test_ds,
    class_names,
    "ResNet50 Fine Tuned"
)