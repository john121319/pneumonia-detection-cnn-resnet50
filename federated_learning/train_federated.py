import os
import csv
import numpy as np
import tensorflow as tf

from model import build_model


# =========================
# Paths
# =========================

CLIENTS_PATH = "../federated_learning/clients_data"
TEST_PATH = "../chest_xray/test"

FINAL_MODEL_PATH = "../results/federated_cnn_final_model.keras"
BEST_MODEL_PATH = "../results/federated_cnn_best_model.keras"
HISTORY_PATH = "../results/federated_training_history.csv"


NUM_CLIENTS = 3
ROUNDS = 5
LOCAL_EPOCHS = 1

BATCH_SIZE = 32
IMG_SIZE = (128, 128)



os.makedirs("../results", exist_ok=True)


def count_client_samples(client_id):

    client_path = os.path.join(
        CLIENTS_PATH,
        f"client_{client_id}"
    )

    valid_extensions = (
        ".jpg",
        ".jpeg",
        ".png",
        ".bmp"
    )

    total_samples = 0

    for class_name in [
        "NORMAL",
        "PNEUMONIA"
    ]:

        class_path = os.path.join(
            client_path,
            class_name
        )

        image_files = [
            file_name
            for file_name in os.listdir(class_path)
            if file_name.lower().endswith(valid_extensions)
        ]

        total_samples += len(
            image_files
        )

    return total_samples

def load_client_dataset(client_id):

    client_path = os.path.join(
        CLIENTS_PATH,
        f"client_{client_id}"
    )

    dataset = tf.keras.utils.image_dataset_from_directory(
        client_path,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        color_mode="grayscale",
        label_mode="binary",
        shuffle=True,
        seed=123
    )

    dataset = dataset.map(
        lambda x, y: (
            tf.cast(x, tf.float32) / 255.0,
            y
        ),
        num_parallel_calls=tf.data.AUTOTUNE
    )

    dataset = dataset.prefetch(
        tf.data.AUTOTUNE
    )

    return dataset

def load_test_dataset():

    test_ds = tf.keras.utils.image_dataset_from_directory(
        TEST_PATH,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        color_mode="grayscale",
        label_mode="binary",
        shuffle=False
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

    return test_ds


def weighted_average_weights(
    client_weights,
    client_sample_counts
):

    total_samples = sum(
        client_sample_counts
    )

    averaged_weights = []

    for layer_weights in zip(*client_weights):

        weighted_sum = np.zeros_like(
            layer_weights[0]
        )

        for weights, sample_count in zip(
            layer_weights,
            client_sample_counts
        ):

            client_ratio = (
                sample_count / total_samples
            )

            weighted_sum += (
                weights * client_ratio
            )

        averaged_weights.append(
            weighted_sum
        )

    return averaged_weights


# =========================
# Evaluate global model
# =========================

def evaluate_global_model(
    model,
    test_ds
):

    results = model.evaluate(
        test_ds,
        verbose=0
    )

    loss = results[0]
    accuracy = results[1]
    precision = results[2]
    recall = results[3]

    return (
        loss,
        accuracy,
        precision,
        recall
    )


global_model = build_model()

test_ds = load_test_dataset()



print("\n====================================")
print("Initial Global Model")
print("====================================")

initial_results = evaluate_global_model(
    global_model,
    test_ds
)

print("Loss:", initial_results[0])
print("Accuracy:", initial_results[1])
print("Precision:", initial_results[2])
print("Recall:", initial_results[3])



round_history = []


best_accuracy = 0.0
best_round = 0



for round_number in range(
    1,
    ROUNDS + 1
):

    print("\n====================================")
    print("Federated Round:", round_number)
    print("====================================")

    global_weights = global_model.get_weights()

    client_weights = []
    client_sample_counts = []



    for client_id in range(
        1,
        NUM_CLIENTS + 1
    ):

        print(
            f"\nTraining client_{client_id}"
        )

        # Build local model
        client_model = build_model()

        # Start from current global weights
        client_model.set_weights(
            global_weights
        )

        # Load local client data
        client_dataset = load_client_dataset(
            client_id
        )

        # Count local samples
        sample_count = count_client_samples(
            client_id
        )

        print(
            "Client samples:",
            sample_count
        )

        # Local training
        client_model.fit(
            client_dataset,
            epochs=LOCAL_EPOCHS,
            verbose=1
        )

        # Save local weights
        client_weights.append(
            client_model.get_weights()
        )

        # Save local sample count
        client_sample_counts.append(
            sample_count
        )



    new_global_weights = weighted_average_weights(
        client_weights,
        client_sample_counts
    )


    global_model.set_weights(
        new_global_weights
    )


    print(
        f"\nEvaluating global model after round {round_number}"
    )

    (
        loss,
        accuracy,
        precision,
        recall
    ) = evaluate_global_model(
        global_model,
        test_ds
    )


    print("\nRound Results")
    print("------------------------------")
    print("Loss:", loss)
    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)



    round_history.append({
        "round": round_number,
        "loss": loss,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall
    })



    if accuracy > best_accuracy:

        best_accuracy = accuracy
        best_round = round_number

        global_model.save(
            BEST_MODEL_PATH
        )

        print(
            "\nNew best global model saved."
        )

        print(
            "Best round:",
            best_round
        )

        print(
            "Best accuracy:",
            best_accuracy
        )


global_model.save(
    FINAL_MODEL_PATH
)

with open(
    HISTORY_PATH,
    mode="w",
    newline=""
) as file:

    writer = csv.DictWriter(
        file,
        fieldnames=[
            "round",
            "loss",
            "accuracy",
            "precision",
            "recall"
        ]
    )

    writer.writeheader()

    writer.writerows(
        round_history
    )


print("\n====================================")
print("Federated Training Completed")
print("====================================")

print(
    "Best round:",
    best_round
)

print(
    "Best accuracy:",
    best_accuracy
)

print(
    "Best model saved to:",
    BEST_MODEL_PATH
)

print(
    "Final model saved to:",
    FINAL_MODEL_PATH
)

print(
    "Training history saved to:",
    HISTORY_PATH
)