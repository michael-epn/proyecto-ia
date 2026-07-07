import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
import tensorflow_datasets as tfds
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score

# 1. Cargar el dataset
dataset, info = tfds.load('oxford_iiit_pet', with_info=True, as_supervised=True)
train_ds, test_ds = dataset['train'], dataset['test']

IMG_SIZE = 128
NUM_CLASSES = 37 
BATCH_SIZE = 32

# 2. Preprocesamiento simple (0 a 1)
def preprocesar(imagen, etiqueta):
    imagen = tf.cast(imagen, tf.float32) / 255.0 
    imagen = tf.image.resize(imagen, (IMG_SIZE, IMG_SIZE))
    return imagen, etiqueta

train_ds = train_ds.map(preprocesar).shuffle(1000).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
test_ds = test_ds.map(preprocesar).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

# 3. TRANSFER LEARNING: Usar MobileNetV2
base_model = MobileNetV2(input_shape=(IMG_SIZE, IMG_SIZE, 3), include_top=False, weights='imagenet')
base_model.trainable = False # Congelamos los pesos pre-entrenados

modelo_custom = models.Sequential([
    layers.InputLayer(input_shape=(IMG_SIZE, IMG_SIZE, 3)),
    # MobileNetV2 espera valores entre [-1, 1], convertimos el [0, 1] a [-1, 1]
    layers.Rescaling(scale=2.0, offset=-1.0), 
    
    # Añadimos el modelo base
    base_model,
    
    # Capas finales de clasificación
    layers.GlobalAveragePooling2D(),
    layers.Dropout(0.3),
    layers.Dense(NUM_CLASSES, activation="softmax")
])

# 4. Compilar y Entrenar (10 épocas son suficientes con Transfer Learning)
modelo_custom.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=["accuracy"])
print("Iniciando entrenamiento con Transfer Learning...")
historial = modelo_custom.fit(train_ds, validation_data=test_ds, epochs=10)

# Guardar en el formato moderno recomendado (.keras)
modelo_custom.save("modelo_pets_custom.keras")

# 5. CALCULAR MÉTRICAS FINALES PARA EL HTML
print("\n--- Calculando Métricas Finales ---")
y_true, y_pred = [], []
for imagenes, etiquetas in test_ds:
    predicciones = modelo_custom.predict(imagenes, verbose=0)
    y_pred.extend(np.argmax(predicciones, axis=1))
    y_true.extend(etiquetas.numpy())

acc = accuracy_score(y_true, y_pred)
prec = precision_score(y_true, y_pred, average='macro', zero_division=0)
rec = recall_score(y_true, y_pred, average='macro', zero_division=0)

print(f"Accuracy: {acc*100:.2f}%")
print(f"Precisión: {prec*100:.2f}%")
print(f"Recall: {rec*100:.2f}%")