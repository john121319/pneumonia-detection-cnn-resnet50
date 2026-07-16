from pathlib import Path
import csv

import numpy as np
import tensorflow as tf

from model import build_model


SEED = 123
tf.keras.utils.set_random_seed(SEED)


FEDERATED_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = FEDERATED_DIR.parent

CLIENTS_PATH = FEDERATED_DIR / "clients_data_non_iid"

TEST_PATH = PROJECT_ROOT / "chest_xray" / "test"

RESULTS_PATH = PROJECT_ROOT / "results"

BEST_MODEL_PATH = (
    RESULTS_PATH
    / "federated_cnn_non_iid_best_model.keras"
)

FINAL_MODEL_PATH = (
    RESULTS_PATH
    / "federated_cnn_non_iid_final_model.keras"
)

HISTORY_PATH = (
    RESULTS_PATH
    / "federated_non_iid_training_history.csv"
)



NUM_CLIENTS = 3
ROUNDS = 5
LOCAL_EPOCHS = 1

BATCH_SIZE = 32
IMG_SIZE = (128, 128)

CLASSES = ["NORMAL", "PNEUMONIA"]

VALID_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp"
}


RESULTS_PATH.mkdir(
    parents=True,
    exist_ok=True
)


def count_client_samples(client_id):

    client_path = CLIENTS_PATH / f"client_{client_id}"

    total_samples = 0

    for class_name in CLASSES:

        class_path = client_path / class_name

        image_files = [
            file_path
            for file_path in class_path.iterdir()
            if (
                file_path.is_file()
                and file_path.suffix.lower() in VALID_EXTENSIONS
            )
        ]

        total_samples += len(image_files)

    return total_samples


def load_client_dataset(client_id):

    client_path = CLIENTS_PATH / f"client_{client_id}"

    dataset = tf.keras.utils.image_dataset_from_directory(
        client_path,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        color_mode="grayscale",
        label_mode="binary",
        shuffle=True,
        seed=SEED
    )

    dataset = dataset.map(
        lambda x, y: (
            tf.cast(x, tf.float32) / 255.0,
            y
        ),
        num_parallel_calls=tf.data.AUTOTUNE
    )

    dataset = dataset.prefetch(tf.data.AUTOTUNE)

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

    test_ds = test_ds.prefetch(tf.data.AUTOTUNE)

    return test_ds


def weighted_average_weights(client_weights, client_sample_counts):

    total_samples = sum(client_sample_counts)

    averaged_weights = []

    for layer_group in zip(*client_weights):

        weighted_sum = np.zeros_like(layer_group[0])

        for client_layer_weights, sample_count in zip(
            layer_group,
            client_sample_counts
        ):

            client_ratio = sample_count / total_samples

            weighted_sum += client_layer_weights * client_ratio

        averaged_weights.append(weighted_sum)

    return averaged_weights


def evaluate_model(model, dataset):

    results = model.evaluate(
        dataset,
        verbose=0,
        return_dict=True
    )

    return results


def save_history(round_history):

    fieldnames = [
        "round",
        "loss",
        "accuracy",
        "precision",
        "recall"
    ]

    with open(
        HISTORY_PATH,
        mode="w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames
        )

        writer.writeheader()
        writer.writerows(round_history)


def main():

    if not CLIENTS_PATH.exists():
        raise FileNotFoundError(
            "Non-IID clients not found. "
            "Run create_non_iid_clients.py first."
        )

    test_ds = load_test_dataset()

    global_model = build_model()

    print("\n====================================")
    print("Initial Non-IID Global Model")
    print("====================================")

    initial_metrics = evaluate_model(
        global_model,
        test_ds
    )

    for name, value in initial_metrics.items():
        print(f"{name}: {value}")

    round_history = []

    best_accuracy = 0.0
    best_round = 0

    for round_number in range(1, ROUNDS + 1):

        print("\n====================================")
        print("Non-IID Federated Round:", round_number)
        print("====================================")

        global_weights = global_model.get_weights()

        client_weights = []
        client_sample_counts = []

        for client_id in range(1, NUM_CLIENTS + 1):

            print(f"\nTraining non-IID client_{client_id}")

            client_model = build_model()

            client_model.set_weights(global_weights)

            client_dataset = load_client_dataset(client_id)

            sample_count = count_client_samples(client_id)

            print("Client samples:", sample_count)

            client_model.fit(
                client_dataset,
                epochs=LOCAL_EPOCHS,
                verbose=1
            )

            client_weights.append(
                client_model.get_weights()
            )

            client_sample_counts.append(
                sample_count
            )

        print("\nAggregating non-IID client weights...")

        new_global_weights = weighted_average_weights(
            client_weights,
            client_sample_counts
        )

        global_model.set_weights(new_global_weights)

        metrics = evaluate_model(
            global_model,
            test_ds
        )

        print("\nRound Results")
        print("------------------------------")

        for name, value in metrics.items():
            print(f"{name}: {value}")

        round_history.append({
            "round": round_number,
            "loss": metrics["loss"],
            "accuracy": metrics["accuracy"],
            "precision": metrics["precision"],
            "recall": metrics["recall"]
        })

        if metrics["accuracy"] > best_accuracy:

            best_accuracy = metrics["accuracy"]
            best_round = round_number

            global_model.save(BEST_MODEL_PATH)

            print("\nNew best Non-IID global model saved.")
            print("Best round:", best_round)
            print("Best accuracy:", best_accuracy)

    global_model.save(FINAL_MODEL_PATH)

    save_history(round_history)

    print("\n====================================")
    print("Non-IID Federated Training Completed")
    print("====================================")

    print("Best round:", best_round)
    print("Best accuracy:", best_accuracy)
    print("Best model:", BEST_MODEL_PATH)
    print("Final model:", FINAL_MODEL_PATH)
    print("History CSV:", HISTORY_PATH)


if __name__ == "__main__":
    main()