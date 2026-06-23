import tensorflow as tf
import numpy as np

model=tf.keras.models.load_model(
    "../results/cnn_pneumonia_detection_model.keras"
)

img = tf.keras.utils.load_img(
    "../chest_xray/test/normal/IM-0001-0001.jpeg",
    target_size=(128, 128),
    color_mode="grayscale"
)

img = tf.keras.utils.img_to_array(img)

img = img / 255.0

img = np.expand_dims(
    img,
    axis=0
)

prediction = model.predict(img)

probability = prediction[0][0]
print("Pneumonia probability:", probability)

if probability > 0.5:
    print("prediction: PNEUMONIA")
else:
    print("prediction: NORMAL")