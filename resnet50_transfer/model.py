import tensorflow as tf

def build_model():
    data_augmentation=tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.1),
        tf.keras.layers.RandomZoom(0.1),
    ])

    base_model=tf.keras.applications.ResNet50(
        weights="imagenet",
        include_top=False,
        input_shape=(224, 224, 3)
    )

    base_model.trainable=False

    inputs=tf.keras.layers.Input(shape=(224, 224, 3))

    x=data_augmentation(inputs)

    x=tf.keras.applications.resnet50.preprocess_input(x)

    x=base_model(x, training=False)
    x=tf.keras.layers.GlobalAveragePooling2D()(x)
    x=tf.keras.layers.Dense(224, activation="relu")(x)
    x=tf.keras.layers.Dropout(0.5)(x)

    outputs=tf.keras.layers.Dense(1, activation="sigmoid")(x)

    model = tf.keras.Model(inputs, outputs)

    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    return model