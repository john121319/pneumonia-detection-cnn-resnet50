from pathlib import Path
import random
import shutil


FEDERATED_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = FEDERATED_DIR.parent

SOURCE_TRAIN_PATH = PROJECT_ROOT / "chest_xray" / "train"
CLIENTS_PATH = FEDERATED_DIR / "clients_data_non_iid"


RANDOM_SEED = 123

VALID_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp"
}


CLASS_DISTRIBUTION = {
    "NORMAL": {
        "client_1": 0.70,
        "client_2": 0.10,
        "client_3": 0.20
    },
    "PNEUMONIA": {
        "client_1": 0.10,
        "client_2": 0.70,
        "client_3": 0.20
    }
}


def get_image_files(folder_path):

    image_files = []

    for file_path in folder_path.iterdir():

        if (
            file_path.is_file()
            and file_path.suffix.lower() in VALID_EXTENSIONS
        ):
            image_files.append(file_path)

    return sorted(image_files)


def create_client_folders():

    if CLIENTS_PATH.exists():
        shutil.rmtree(CLIENTS_PATH)

    for client_name in ["client_1", "client_2", "client_3"]:

        for class_name in ["NORMAL", "PNEUMONIA"]:

            folder_path = (
                CLIENTS_PATH
                / client_name
                / class_name
            )

            folder_path.mkdir(
                parents=True,
                exist_ok=True
            )

def split_class_non_iid(class_name):

    class_path = SOURCE_TRAIN_PATH / class_name

    if not class_path.exists():
        raise FileNotFoundError(
            f"Class folder not found: {class_path}"
        )

    images = get_image_files(class_path)

    random.shuffle(images)

    total_images = len(images)

    print("\n====================================")
    print("Class:", class_name)
    print("Total images:", total_images)
    print("====================================")

    start_index = 0

    client_names = list(
        CLASS_DISTRIBUTION[class_name].keys()
    )

    for index, client_name in enumerate(client_names):

        ratio = CLASS_DISTRIBUTION[class_name][client_name]

        if index == len(client_names) - 1:
            end_index = total_images
        else:
            image_count = int(total_images * ratio)
            end_index = start_index + image_count

        client_images = images[start_index:end_index]

        destination_folder = (
            CLIENTS_PATH
            / client_name
            / class_name
        )

        for source_file in client_images:

            destination_file = (
                destination_folder
                / source_file.name
            )

            shutil.copy2(
                source_file,
                destination_file
            )

        print(
            client_name,
            ":",
            len(client_images),
            class_name,
            "images"
        )

        start_index = end_index


def count_images(folder_path):

    return len(
        get_image_files(folder_path)
    )


def print_client_summary():

    print("\n====================================")
    print("Non-IID Client Summary")
    print("====================================")

    grand_total = 0

    for client_name in ["client_1", "client_2", "client_3"]:

        print(f"\n{client_name}")

        client_total = 0

        for class_name in ["NORMAL", "PNEUMONIA"]:

            folder_path = (
                CLIENTS_PATH
                / client_name
                / class_name
            )

            count = count_images(folder_path)

            client_total += count

            print(class_name, ":", count)

        grand_total += client_total

        print("Total:", client_total)

    print("\nGrand total:", grand_total)


def main():

    random.seed(RANDOM_SEED)

    print("Source:", SOURCE_TRAIN_PATH)
    print("Output:", CLIENTS_PATH)

    create_client_folders()

    split_class_non_iid("NORMAL")
    split_class_non_iid("PNEUMONIA")

    print_client_summary()

    print("\nNon-IID clients created successfully.")


if __name__ == "__main__":
    main()