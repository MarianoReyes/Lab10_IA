''' 
TASK 1.3
'''

import tensorflow as tf
import cv2
import numpy as np

img_size = 100

img = cv2.imread("gato_prueba.jpg")
img = cv2.resize(img, (img_size, img_size))
img = np.expand_dims(img, axis=0)

model = tf.keras.models.load_model("modelo.h5")
prediction = model.predict(img)

if prediction < 0.5:
    print("La imagen es de un perro.")
else:
    print("La imagen es de un gato.")
