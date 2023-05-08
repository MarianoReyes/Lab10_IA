import tensorflow as tf
import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split

'''
TASK 1.1
'''

data_path = "PetImages/"
categories = ["Dog", "Cat"]
img_size = 100

# Leer imágenes y etiquetas
data = []
for category in categories:
    path = os.path.join(data_path, category)
    label = categories.index(category)
    for img_name in os.listdir(path):
        img_path = os.path.join(path, img_name)
        img = cv2.imread(img_path)
        if img is None:
            continue
        if img.size == 0:
            continue
        img = cv2.resize(img, (img_size, img_size))
        data.append([img, label])

# Convertir a arreglo numpy y separar en entrenamiento y prueba
np.random.shuffle(data)
X = np.array([item[0] for item in data])
y = np.array([item[1] for item in data])
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)


'''
TASK 1.2
'''

# crear modelo
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu',
                           input_shape=X_train.shape[1:]),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# compilar modelo
model.compile(optimizer='adam', loss='binary_crossentropy',
              metrics=['accuracy'])

# entrenar el modelo
epochs = 10
batch_size = 32
history = model.fit(X_train, y_train, epochs=epochs,
                    batch_size=batch_size, validation_split=0.1)

# evaluar el modelo
train_loss, train_acc = model.evaluate(X_train, y_train, verbose=2)
test_loss, test_acc = model.evaluate(X_test, y_test)

print("Train accuracy:", train_acc)
print('Test accuracy:', test_acc)

# Guardar modelo
model.save("modelo.h5")
