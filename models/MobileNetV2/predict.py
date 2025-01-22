import os

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from keras.api.preprocessing import image
from keras.src.applications.mobilenet_v2 import preprocess_input

# Załadowanie modelu zapisanego po treningu
model = tf.keras.models.load_model('mobilenetv2_katakana_model.keras')

# Funkcja do rozpoznawania obrazu
def predict_katakana(image_path):
    img = image.load_img(image_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    predictions = model.predict(img_array)

    predicted_class = np.argmax(predictions, axis=1)[0]
    confidence = np.max(predictions, axis=1)[0]

    class_names = os.listdir('../../datasets/dataset_mnv2/train')

    predicted_class_name = class_names[predicted_class]

    print(f"Predicted Class: {predicted_class_name}")
    print(f"Confidence: {confidence:.4f}")

    # Wyświetlanie obrazu
    plt.imshow(img)
    plt.axis('off')
    plt.show()

image_path = '../../images/ta.png'
predict_katakana(image_path)
