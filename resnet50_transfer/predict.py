import tensorflow as tf
import numpy as np




MODEL_PATH = "../results/resnet50_pneumonia_finetuned_model.keras"
IMAGE_PATH = "../chest_xray/test/IM-0019-0001.jpeg"




model = tf.keras.models.load_model(MODEL_PATH)



img = tf.keras.utils.load_img(
    IMAGE_PATH,
    target_size=(224, 224),
    color_mode="rgb"
)



img = tf.keras.utils.img_to_array(img)



img = np.expand_dims(img, axis=0)


print("Image shape:", img.shape)



prediction = model.predict(img)

probability = prediction[0][0]

print("Pneumonia probability:", probability)



if probability > 0.5:
    print("Prediction: PNEUMONIA")
else:
    print("Prediction: NORMAL")