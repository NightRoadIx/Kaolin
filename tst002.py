# Este se trata de un problema de regresión para obtener los valores de viscosidad del Caolín Blanco o Beige
# de acuerdo a una serie de datos
# El objetivo de es predecir la salida de un valor continuo.
# Se va a utilizar la API tf.keras

# Keras es una API de alto nivel de redes neuronales
# https://keras.io/
# https://www.tensorflow.org/guide/keras

# Debe instalarse el paquete seaborn para la graficación por pares
# pip install -q seaborn

from __future__ import absolute_import, division, print_function, unicode_literals

import pathlib

# Graficación y manejo de archivos CSV
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Indispensable para utilizar keras
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

print(tf.version.VERSION)
print(tf.__version__)

#%%
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# INGRESO DE DATOS
# Colocar los nombres de las columnas
column_names = ['N','k','mu','alfa','tau', 'gamma', 'pH', 'Cw', 'Blanco', 'Beige']
raw_dataset = pd.read_csv("caolin.csv", names=column_names, na_values = "?", comment='\t', sep=",", skipinitialspace=True)

#%%
# Copiar los datos de la tabla en una nueva variable
dataset = raw_dataset.copy()
# Mostrar los últimos datos
dataset.tail()

# Verificar si hay datos NA de acuerdo a las columnas que se colocaron antes
dataset.isna().sum()

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# ENTRENAMIENTO (PREPARACIÓN DE DATOS)
# Primero lo que se hace es dividir los datos en:
# Datos de entrenamiento
train_dataset = dataset.sample(frac=0.8,random_state=None)
# Datos de prueba
test_dataset = dataset.drop(train_dataset.index)

# Usando seaborn, se pueden obtener gráficas de del grupo de datos de entrenamiento
sns.pairplot(train_dataset[['mu','alfa','tau', 'gamma', 'pH', 'Cw']], diag_kind="kde")
plt.show()

# También se da una vista rápida a las estadísticas principales
train_stats = train_dataset.describe()
train_stats.pop("mu")
train_stats = train_stats.transpose()
train_stats

# Dividir las características de las etiquetas
# Separar el valor objetivo, o "etiqueta", de las características. ESta etiqueta es el valor que será entrenado para predecir el modelo
train_labels = train_dataset.pop('mu')
test_labels = test_dataset.pop('mu')

# Normalizar los datos 
# Debido a que los intervalos de los datos pueden presentar grandes variaciones, es una buena práctica el normalizar las características
# que usan diferentes escalas e intervalos. A pesar de que el modelo puede converger sin la normalización de las características, 
# hará que el entrenamiento sea más complicado, además de que el modelo resultante dependerá de la elección de las unidades
# utilizadas en la entrada

# Se genera una función de normalización
def norm(x): return(x - train_stats['mean']) / train_stats['std']

# Normalizar tanto los datos de entrenamiento como de prueba
normed_train_data = norm(train_dataset)
normed_test_data = norm(test_dataset)

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# MODELO
# Crear el modelo
# Aquí se utiliza un modelo Sequential con dos capas ocultas densamente conectadas, y una capa de salida  que regresa un valor
# continuo. Los pasos para la construcción del modelo están colocados en una función. 
def build_model():
 model = keras.Sequential(
         [layers.Dense(9, activation=tf.nn.relu, input_shape=[len(train_dataset.keys())]), layers.Dense(64, activation=tf.nn.relu), layers.Dense(1)])
 optimizer = tf.keras.optimizers.RMSprop(0.001)
 model.compile(loss='mean_squared_error', optimizer=optimizer, metrics=['mean_absolute_error', 'mean_squared_error'])
 return model

# Generar el modelo
# TODO: Aquí se genera una advertencia de obsolesencia
# Call initializer instance with the dtype argument instead of passing it to the constructor
model = build_model()

# Inspeccionar el modelo
model.summary()

# Probar el modelo con una pequeña muestra de los datos de prueba
example_batch = normed_train_data[:10]
example_result = model.predict(example_batch)
example_result

# Entrenar el modelo
# Desplegar el proceso de entrenamiento al colocar un punto para cada época completada
class PrintDot(keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs):
    if epoch % 100 == 0: print('')
    print('.', end='')

# Número de epocas a tratar
EPOCHS = 1000

history = model.fit(
  normed_train_data, train_labels,
  epochs=EPOCHS, validation_split = 0.2, verbose=0,
  callbacks=[PrintDot()])
  
# Visualizar el progreso del entrenamiento usando las estadísticas de entrenamiento del proceso usadas/almacenadas en el objeto history
hist = pd.DataFrame(history.history)
hist['epoch'] = history.epoch
hist.tail()

# Graficar el error que se obtiene del modelo con cada epoca/iteración
def plot_history(history):
 hist = pd.DataFrame(history.history)
 hist['epoch'] = history.epoch
 plt.figure()
 plt.xlabel('Epoca')
 plt.ylabel('Error Abs Prom [mu]')
 plt.plot(hist['epoch'], hist['mean_absolute_error'], label='Train Error')
 plt.plot(hist['epoch'], hist['val_mean_absolute_error'], label = 'Val Error')
 plt.ylim([0,5])
 plt.legend()
 plt.figure()
 plt.xlabel('Epoca')
 plt.ylabel('Error Cuadr. Medio [$mu^2$]')
 plt.plot(hist['epoch'], hist['mean_squared_error'], label='Train Error')
 plt.plot(hist['epoch'], hist['val_mean_squared_error'], label = 'Val Error')
 plt.ylim([0,20])
 plt.legend()
 plt.show()

# Mandar las gráficas del error, con esto se puede observar si el modelo muestra alguna mejora o incluso presenta degradación
# con cada época/iteración
plot_history(history)

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# MEJORA DEL MODELO
# Si el modelo no muestra mejora, lo que se hace es utilizar un EarlyStopping callback, el cual probará una condición de entrenamiento
# para cada epoca. Si un número de epocas no muestra mejora, entonces detiene el entrenamiento
model = build_model()

# El parámetro de paciencia es la cantidad de épocas para revisar en la mejora
early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)
history = model.fit(normed_train_data, train_labels, epochs=EPOCHS, validation_split = 0.2, verbose=0, callbacks=[early_stop, PrintDot()])
plot_history(history)

# Aquí se observa que tan bien el modelo generaliza al utilizar el grupo de prueba, el cual por supuesto no se utilizo para entrenar el modelo
# Esto indica que tan bien esperamos que el modelo realice predicciones en el mundo real
loss, mae, mse = model.evaluate(normed_test_data, test_labels, verbose=0)
print("Grupo de prueba MAE: {:5.2f} mu".format(mae))

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# PREDICCIONES
# Finalmente, realizar predicciones de los valores de viscosidad en el grupo de prueba
test_predictions = model.predict(normed_test_data).flatten()

plt.scatter(test_labels, test_predictions, color="blue")
plt.xlabel('Valores reales \u03BC [mPa s]')
plt.ylabel('Predicciones \u03BC [mPa s]')
plt.axis('equal')
plt.axis('square')
plt.xlim([0,plt.xlim()[1]])
plt.ylim([0,plt.ylim()[1]])
_ = plt.plot([0, 1750], [0, 1750], color="red")
plt.show()

# Distribución de los errores
error = test_predictions - test_labels
plt.hist(error, bins = 25)
plt.xlabel("Error de Predicción[mu]")
_ = plt.ylabel("Cuentas")
plt.show()


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# GUARDAR DATOS A ARCHIVO
# Mandar datos a un archivo csv
#import csv
#with open('people.csv', 'w') as writeFile:
# writer = csv.writer(writeFile)
# writer.writerows(list(test_predictions))

import numpy
a = numpy.asarray(list(test_predictions))

pd.read_csv("caolin.csv", names=column_names, na_values = "?", comment='\t', sep=",", skipinitialspace=True)
test_dataset.to_csv('prueba.csv',sep=",")
