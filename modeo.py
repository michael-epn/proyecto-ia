import tensorflow as tf
from tensorflow.keras import layers, models
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt


# 1. Cargar el dataset directamente desde la librería tfds
dataset, info = tfds.load('oxford_iiit_pet', with_info=True, as_supervised=True)
train_ds, test_ds = dataset['train'], dataset['test']

# 2. Preprocesamiento: Normalización y Redimensionamiento
IMG_SIZE = 128
NUM_CLASSES = 37 # El dataset tiene 37 razas de mascotas
BATCH_SIZE = 32

def preprocesar(imagen, etiqueta):
    imagen = tf.cast(imagen, tf.float32) / 255.0 # Estandarizar entre 0 y 1
    imagen = tf.image.resize(imagen, (IMG_SIZE, IMG_SIZE)) # Tamaño uniforme
    return imagen, etiqueta

# Aplicar preprocesamiento, crear lotes y optimizar rendimiento
train_ds = train_ds.map(preprocesar).shuffle(1000).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
test_ds = test_ds.map(preprocesar).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

# 3. Crear Arquitectura (CNN Mejorada)
modelo_custom = models.Sequential([
    # Data Augmentation (crea variaciones de las imágenes para que el modelo aprenda mejor)
    layers.RandomFlip("horizontal", input_shape=(IMG_SIZE, IMG_SIZE, 3)),
    layers.RandomRotation(0.1),
    
    # Bloque Convolucional 1
    layers.Conv2D(32, (3,3), padding='same', activation="relu"),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2,2)),
    
    # Bloque Convolucional 2
    layers.Conv2D(64, (3,3), padding='same', activation="relu"),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2,2)),
    
    # Bloque Convolucional 3
    layers.Conv2D(128, (3,3), padding='same', activation="relu"),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2,2)),
    
    # Aplanamiento y Capas Densas
    layers.Flatten(),
    layers.Dense(256, activation="relu"),
    layers.Dropout(0.5), # Apaga el 50% de neuronas al azar para evitar memorización
    layers.Dense(NUM_CLASSES, activation="softmax")
])

# 4. Compilar (usamos sparse porque las etiquetas son enteros, no one-hot)
modelo_custom.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=["accuracy"])

# 5. Entrenar (Aumentamos épocas porque el problema es más difícil)
historial = modelo_custom.fit(train_ds, validation_data=test_ds, epochs=20)

modelo_custom.save("modelo_pets_custom.h5")