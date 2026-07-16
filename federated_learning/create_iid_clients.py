import os
import shutil
import random



SOURCE_TRAIN_PATH = "../chest_xray/train"
CLIENTS_PATH = "../federated_learning/clients_data"



NUM_CLIENTS = 3
CLASSES = ["NORMAL", "PNEUMONIA"]
RANDOM_SEED = 123



def create_client_folders():

    if os.path.exists(CLIENTS_PATH):
        shutil.rmtree(CLIENTS_PATH)

    for client_id in range(1, NUM_CLIENTS + 1):
        for class_name in CLASSES:
            folder_path = os.path.join(
                CLIENTS_PATH,
                f"client_{client_id}",
                class_name
            )

            os.makedirs(folder_path, exist_ok=True)



def split_data_among_clients():

    random.seed(RANDOM_SEED)

    for class_name in CLASSES:

        class_path = os.path.join(
            SOURCE_TRAIN_PATH,
            class_name
        )

        images = os.listdir(class_path)

        random.shuffle(images)

        total_images = len(images)

        images_per_client = total_images // NUM_CLIENTS

        print("\nClass:", class_name)
        print("Total images:", total_images)

        for client_id in range(1, NUM_CLIENTS + 1):

            start_index = (client_id - 1) * images_per_client

            if client_id == NUM_CLIENTS:
                end_index = total_images
            else:
                end_index = client_id * images_per_client

            client_images = images[start_index:end_index]

            destination_folder = os.path.join(
                CLIENTS_PATH,
                f"client_{client_id}",
                class_name
            )

            for image_name in client_images:

                source_file = os.path.join(
                    class_path,
                    image_name
                )

                destination_file = os.path.join(
                    destination_folder,
                    image_name
                )

                shutil.copy2(
                    source_file,
                    destination_file
                )

            print(
                f"client_{client_id}:",
                len(client_images),
                class_name,
                "images"
            )


def check_clients():

    print("\n==============================")
    print("Client Data Summary")
    print("==============================")

    for client_id in range(1, NUM_CLIENTS + 1):

        print(f"\nclient_{client_id}")

        for class_name in CLASSES:

            folder_path = os.path.join(
                CLIENTS_PATH,
                f"client_{client_id}",
                class_name
            )

            count = len(os.listdir(folder_path))

            print(class_name, ":", count)



create_client_folders()
split_data_among_clients()
check_clients()

print("\nFederated client data created successfully.")