import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt



MODEL_PATH = "../results/resnet50_pneumonia_finetuned_model.keras"
IMAGE_PATH = "../chest_xray/test/PNEUMONIA/person1_virus_6.jpeg"

ORIGINAL_OUTPUT_PATH = "../results/pneumonia_gradcam_original.png"
HEATMAP_OUTPUT_PATH = "../results/pneumonia_gradcam_heatmap.png"
OVERLAY_OUTPUT_PATH = "../results/pneumonia_gradcam_output.png"



IMG_SIZE = (224, 224)



os.makedirs("../results", exist_ok=True)



def load_image(image_path):

    img = tf.keras.utils.load_img(
        image_path,
        target_size=IMG_SIZE,
        color_mode="rgb"
    )

    img_array = tf.keras.utils.img_to_array(img)

    img_array = np.expand_dims(
        img_array,
        axis=0
    )

    return img, img_array



def build_gradcam_model_from_trained_model(trained_model):

    trained_resnet = trained_model.get_layer("resnet50")
    trained_dense = trained_model.get_layer("dense")
    trained_output = trained_model.get_layer("dense_1")

    base_model = tf.keras.applications.ResNet50(
        weights=None,
        include_top=False,
        input_shape=(224, 224, 3)
    )

    base_model.set_weights(
        trained_resnet.get_weights()
    )

    dense_layer = tf.keras.layers.Dense(
        224,
        activation="relu"
    )

    output_layer = tf.keras.layers.Dense(
        1,
        activation="sigmoid"
    )

    dense_layer.build(
        (None, 2048)
    )

    output_layer.build(
        (None, 224)
    )

    dense_layer.set_weights(
        trained_dense.get_weights()
    )

    output_layer.set_weights(
        trained_output.get_weights()
    )

    last_conv_layer = base_model.get_layer(
        "conv5_block3_out"
    )

    feature_model = tf.keras.Model(
        inputs=base_model.input,
        outputs=[
            last_conv_layer.output,
            base_model.output
        ]
    )

    return feature_model, dense_layer, output_layer



def make_gradcam_heatmap(
    feature_model,
    dense_layer,
    output_layer,
    img_array
):

    img_array = tf.keras.applications.resnet50.preprocess_input(
        img_array
    )

    output_weights, output_bias = output_layer.get_weights()

    with tf.GradientTape() as tape:

        conv_outputs, features = feature_model(
            img_array
        )

        x = tf.keras.layers.GlobalAveragePooling2D()(
            features
        )

        x = dense_layer(
            x
        )

        logits = tf.matmul(
            x,
            output_weights
        ) + output_bias

        probabilities = tf.sigmoid(
            logits
        )

        probability = probabilities[:, 0]

        if probability.numpy()[0] > 0.5:
            class_score = logits[:, 0]
        else:
            class_score = -logits[:, 0]

    gradients = tape.gradient(
        class_score,
        conv_outputs
    )

    pooled_gradients = tf.reduce_mean(
        gradients,
        axis=(0, 1, 2)
    )

    conv_outputs = conv_outputs[0]

    heatmap = conv_outputs @ pooled_gradients[..., tf.newaxis]

    heatmap = tf.squeeze(
        heatmap
    )

    heatmap = tf.maximum(
        heatmap,
        0
    )

    max_value = tf.reduce_max(
        heatmap
    )

    print("Raw heatmap max:", max_value.numpy())

    if max_value == 0:
        print("Warning: heatmap is zero. Trying absolute heatmap.")

        heatmap = tf.abs(
            conv_outputs @ pooled_gradients[..., tf.newaxis]
        )

        heatmap = tf.squeeze(
            heatmap
        )

        max_value = tf.reduce_max(
            heatmap
        )

    if max_value == 0:
        print("Warning: heatmap is still zero.")
        return heatmap.numpy(), probability.numpy()[0]

    heatmap = heatmap / max_value

    print("Heatmap min:", np.min(heatmap.numpy()))
    print("Heatmap max:", np.max(heatmap.numpy()))

    return heatmap.numpy(), probability.numpy()[0]



def save_gradcam_images(
    original_img,
    heatmap
):

    original_img = np.array(
        original_img
    ).astype("uint8")

    heatmap_resized = tf.image.resize(
        heatmap[..., np.newaxis],
        (
            original_img.shape[0],
            original_img.shape[1]
        )
    )

    heatmap_resized = tf.squeeze(
        heatmap_resized
    ).numpy()

    plt.figure(figsize=(8, 8))
    plt.imshow(original_img)
    plt.axis("off")
    plt.savefig(
        ORIGINAL_OUTPUT_PATH,
        bbox_inches="tight",
        pad_inches=0
    )
    plt.close()

    plt.figure(figsize=(8, 8))
    plt.imshow(
        heatmap_resized,
        cmap="jet"
    )
    plt.axis("off")
    plt.savefig(
        HEATMAP_OUTPUT_PATH,
        bbox_inches="tight",
        pad_inches=0
    )
    plt.close()

    plt.figure(figsize=(8, 8))
    plt.imshow(original_img)
    plt.imshow(
        heatmap_resized,
        cmap="jet",
        alpha=0.6
    )
    plt.axis("off")
    plt.savefig(
        OVERLAY_OUTPUT_PATH,
        bbox_inches="tight",
        pad_inches=0
    )
    plt.close()



trained_model = tf.keras.models.load_model(
    MODEL_PATH
)

original_img, img_array = load_image(
    IMAGE_PATH
)

feature_model, dense_layer, output_layer = build_gradcam_model_from_trained_model(
    trained_model
)

heatmap, probability = make_gradcam_heatmap(
    feature_model,
    dense_layer,
    output_layer,
    img_array
)

save_gradcam_images(
    original_img,
    heatmap
)

print("Pneumonia probability:", probability)

if probability > 0.5:
    print("Prediction: PNEUMONIA")
else:
    print("Prediction: NORMAL")

print("Original image saved to:", ORIGINAL_OUTPUT_PATH)
print("Heatmap image saved to:", HEATMAP_OUTPUT_PATH)
print("Overlay image saved to:", OVERLAY_OUTPUT_PATH)