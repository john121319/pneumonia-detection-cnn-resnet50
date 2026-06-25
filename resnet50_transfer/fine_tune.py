import os
import tensorflow as tf
from model import build_model

TRAIN_PATH = "../chest_xray/train"
VAL_PATH="../chest_xray/val"
TEST_PATH = "../chest_xray/test"

BASE_MODEL_PATH = "../results/resnet50_pneumonia_augmented_model.keras"
FINE_TUNED_MODEL_PATH = "../results/resnet50_pneumonia_finetuned_model.keras"

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
FINE_TUNE_EPOCHS = 5
LEARNING_RATE = 1e-5
UNFREEZE_LAST_N_LAYERS = 30

os.makedirs("../results", exist_ok=True)

train_ds=tf.keras.utils.image_dataset_from_directory(
    TRAIN_PATH,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    color_mode="rgb",
    label_mode="binary",
    shuffle=True
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    VAL_PATH,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    color_mode="rgb",
    label_mode="binary",
    shuffle=False
)

test_ds = tf.keras.utils.image_dataset_from_directory(
    TEST_PATH,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    color_mode="rgb",
    label_mode="binary",
    shuffle=False
)

AUTOTUNE=tf.data.AUTOTUNE

train_ds=train_ds.prefetch(AUTOTUNE)
val_ds = val_ds.prefetch(AUTOTUNE)
test_ds = test_ds.prefetch(AUTOTUNE)

model=tf.keras.models.load_model(BASE_MODEL_PATH)

base_model=model.get_layer('resnet50')

base_model.trainable=True


for layer in base_model.layers[:-UNFREEZE_LAST_N_LAYERS]:
    layer.trainable=False

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
    loss="binary_crossentropy",
    metrics=[
        "accuracy",
        tf.keras.metrics.Precision(name="precision"),
        tf.keras.metrics.Recall(name="recall")
    ]
)


model.summary()

early_stop = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=3,
    restore_best_weights=True
)

checkpoint = tf.keras.callbacks.ModelCheckpoint(
    FINE_TUNED_MODEL_PATH,
    monitor="val_recall",
    save_best_only=True,
    mode="max"
)

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=FINE_TUNE_EPOCHS,
    callbacks=[early_stop, checkpoint]
)


results = model.evaluate(test_ds)

for name, value in zip(model.metrics_names, results):
    print(f"{name}: {value}")

model.save(FINE_TUNED_MODEL_PATH)
