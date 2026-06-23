import tensorflow as tf
from model import build_model

IMG_SIZE = (128, 128)
BATCH_SIZE = 32

train_ds = tf.keras.utils.image_dataset_from_directory(
    "../chest_xray/train",
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    color_mode="grayscale",
    label_mode="binary",
    shuffle=True
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    "../chest_xray/val",
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    color_mode="grayscale",
    label_mode="binary",
    shuffle=True
)

test_ds = tf.keras.utils.image_dataset_from_directory(
    "../chest_xray/test",
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    color_mode="grayscale",
    label_mode="binary",
    shuffle=True
)

normalization_layer = tf.keras.layers.Rescaling(1./255)

train_ds = train_ds.map(
    lambda x,y: (normalization_layer(x), y)
)

val_ds = val_ds.map(
    lambda x,y: (normalization_layer(x), y)
)

test_ds = test_ds.map(
    lambda x,y: (normalization_layer(x), y)
)

AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.prefetch(AUTOTUNE)
val_ds = val_ds.prefetch(AUTOTUNE)
test_ds = test_ds.prefetch(AUTOTUNE)

model = build_model()

model.summary()

early_stop = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=3,
    restore_best_weights=True
)


checkpoint = tf.keras.callbacks.ModelCheckpoint(
    "../results/cnn_pneumonia_detection_model.keras",
    monitor="val_recall",
    save_best_only=True,
    mode="max"
)

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=10
)

loss, accuracy, precision, recall = model.evaluate(test_ds)

print("CNN Test Accuracy:", accuracy)
print("CNN Test Precision:", precision)
print("CNN Test Recall:", recall)
print("CNN Test Loss:", loss)

model.save(
    "../results/cnn_pneumonia_detection_model.keras"
)