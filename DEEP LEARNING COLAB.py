import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt

#Cargar datos de entranamiento y pruebas (imagenes de números)
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

#ver imagen
plt.imshow(x_train[1000])

#concer tamaño y numero de imagenes
x_train.shape

#Preparar datos
#Estandarizar los valores entre 0 y 1 de pixeles (0 255)
x_train=x_train.astype("float32")/255
x_test=x_test.astype("float32")/255

#Cambiar forma de los datos (-1: establece la dimension, tamño pixeles, b/n(1) o color(3))
x_train=x_train.reshape(-1,28,28,1)
x_test=x_test.reshape(-1,28,28,1)

#Crear Arquitectura de la red neuronal convolucional
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
#capa convolucion(#filtros, kernel,fun actvación, forma entrada)
capa_convolucion=Conv2D(32,kernel_size=(3,3),activation="relu",input_shape=(28,28,1))
#Capa de agrupamiento, cada región de 2x2 píxeles en la entrada se reduce a un solo píxe
capa_agrupamiento=MaxPooling2D(pool_size=(2,2))
#Cpa de aplanamiento
capa_aplanamiento=Flatten()
#Caps Densas ocultas (numero de nuronas)
capa_oculta= Dense(units=64,activation="relu")
#capa salida
capa_salida=Dense(units=10,activation="softmax")

modelo=keras.Sequential([capa_convolucion,capa_agrupamiento,capa_aplanamiento, capa_oculta, capa_salida])

#Compilar el modelo
modelo.compile(optimizer='adam', loss='categorical_crossentropy',metrics=["accuracy"])

#pasar a categoricas (10 salidas)
y_train = keras.utils.to_categorical(y_train, num_classes=10)
y_test = keras.utils.to_categorical(y_test, num_classes=10)

#Entrenar el modelo
modelo.fit(x_train,y_train,batch_size=128, epochs=5)

# Guardar la arquitectura, pesos y configuración del optimizador
modelo.save("modelo_mnist.h5")