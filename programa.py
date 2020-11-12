# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, unicode_literals
import sys

# # #Importar aquí las librerías a utilizar # # #
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

# Indispensable para utilizar keras y TensorFlow
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Librerías del PyQt
from PyQt5 import uic, QtWidgets, QtCore, QtGui
from PyQt5.QtWidgets import QApplication, QWidget, QCheckBox, QInputDialog, QLineEdit, QFileDialog, QTableWidget, QTableWidgetItem, QVBoxLayout
from PyQt5.QtGui import QIcon
from PyQt5.QtCore import Qt, pyqtSlot

from matplotlib.backends.backend_qt5agg import (NavigationToolbar2QT as NavigationToolbar)

# Nombre del archivo
qtCreatorFile = "ventagui.ui"

Ui_MainWindow, QtBaseClass = uic.loadUiType(qtCreatorFile)

# # # Funciones para ajuste de curvas # # #
# Función lineal
def lineal(x, a, b):
    return a*x + b    # f(x) = ax + b
# Función exponencial
def exponencial(x, a, b):
    return a*np.exp(b*x)
# Función de ecuación de potencias 
def potencias(x, a, b):
    return a*np.power(x, b)
# Función gaussiana
def gaussiana(x, a, b, c):
    return a * np.exp(-np.power(x - b, 2)/(2*np.power(c, 2)))
# Función sigmoidal
def sigmoide(x, L ,x0, k, b):
    y = L / (1 + np.exp(-k*(x - x0))) + b
    return y

# Cálculo de la suma de los residuos de los cuadrados para le ajuste
def obtenerRCuadrada(xdata, ydata, pars):
    # Obtener la suma de residuos de los cuadrados
    residuos = ydata - potencias(xdata, *pars)
    ss_res = np.sum(residuos**2)
    # Obtener la suma total de cuadrados
    ss_tot = np.sum((ydata - np.mean(ydata))**2)
    # Finalmente, hallar R^2
    r_cuad = 1 - (ss_res / ss_tot)
    return r_cuad
        
# Clase principal
class MyApp(QtWidgets.QMainWindow, Ui_MainWindow):
    # Constructor de la ventana
    def __init__(self):
        QtWidgets.QMainWindow.__init__(self)
        Ui_MainWindow.__init__(self)
        self.setupUi(self)
        
        # Título de la ventana e ícono
        self.title = "Caolín"
        self.setWindowTitle(self.title)
        self.setWindowIcon(QtGui.QIcon('icono.ico'))
        
        # Controladores
        # Habilitar / Deshabilitar controladores
        self.lineaArchivo.setEnabled(False)
        self.caolinPct.setEnabled(False)
        self.caolinPH.setEnabled(False)
        self.caolinPct.setEnabled(False)
        self.caolinTipo.setEnabled(False)
        self.graficar.setEnabled(False)
        self.ajustar.setEnabled(False)
        self.entrenarModelo.setEnabled(False)
        self.predecirModelo.setEnabled(False)
        self.epok.setEnabled(False)
        self.autoEpok.setEnabled(False)
        self.label_6.setVisible(False)
        
        # Conectar acciones de Widgets con funciones de la clase
        self.botonAbrir.clicked.connect(self.openFileNameDialog)
        self.caolinPct.currentTextChanged.connect(self.selectPct)
        self.caolinPH.currentTextChanged.connect(self.selectPH)
        #self.caolinTipo.currentTextChanged.connect(self.final)
        self.graficar.clicked.connect(self.graficarDatos)
        self.ajustar.clicked.connect(self.ajustarDatos)
        self.entrenarModelo.clicked.connect(self.entrenarRedDatos)
        self.autoEpok.stateChanged.connect(self.cambiarAutoAjuste)
        
        # Mandar a la consola la versión del TensorFlow
        print(tf.version.VERSION)
        print(tf.__version__)
                        
        # Agregar la ToolBar para graficar
        #self.addToolBar(NavigationToolbar(self.MplWidget.canvas, self))
        
    # MÉTODOS DE LA CLASE
    # Método para Abrir el archivo CSV
    def openFileNameDialog(self):
        # Crear un cuadro de diálogo para recibir la ruta del archivo
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        fileName, _ = QFileDialog.getOpenFileName(self,"QFileDialog.getOpenFileName()", "","Archivos CSV (*.csv);;Todos los archivos (*)", options=options)
        # Una vez que se tenga un valor
        if fileName:
            # UNDONE: Eliminar esta prueba
            print(fileName)
            # Subir el nombre del archivo (ruta completa)
            self.lineaArchivo.setText(fileName)
            # Abrir mediante pandas el archivo CSV (ajustar apertura con lo que TF requiere)
            self.df = pd.read_csv(fileName, na_values = "?", comment='\t', sep=",", skipinitialspace=True)
            # Cargar los datos en el primer combobox
            # TODO: Modificar para cambiar a porcentaje
            pcts = [i * 100 for i in self.df["Cw"].unique().tolist()]
            print(pcts)
            
            # Cargar los datos en el combobox
            self.caolinPct.addItems(self.df["Cw"].unique().astype(str).tolist())
            # Activar el combobox del porcentaje
            self.caolinPct.setEnabled(True)
            
            # Activar el cuadro de entrenamiento
            self.entrenarModelo.setEnabled(True)
            self.epok.setEnabled(True)
            self.autoEpok.setEnabled(True)
            self.lineaArchivo_2.setEnabled(True)
    
    # Método al seleccionar el porcentaje
    def selectPct(self):
        # Ya que se tiene el porcentaje seleccionado, 
        # cargar el siguiente combobox
        # Hallar los datos del porcentaje seleccionado
        self.pctSelecc = self.df[self.df["Cw"] == float(self.caolinPct.currentText())]
        print(self.pctSelecc["pH"].unique().astype(str).tolist())
        self.caolinPH.clear()
        self.caolinPH.addItems(self.pctSelecc["pH"].unique().astype(str).tolist())
        self.caolinPH.setEnabled(True)
                   
    # Método al seleccionar el PH
    def selectPH(self):
        # Ya que se tiene el PH seleccionado, 
        # cargar el siguiente combobox
        # Hallar los datos del porcentaje seleccionado
        listaTipos = ["Blanco", "Beige"]
        self.caolinTipo.clear()
        self.caolinTipo.addItems(listaTipos)
        self.caolinTipo.setEnabled(True)
        self.graficar.setEnabled(True)
        self.ajustar.setEnabled(True)
        
    # Método para graficar los datos
    def graficarDatos(self):
        # Generar la lista de datos a graficar
        # primero se obtiene la lista de pH
        conPH = self.pctSelecc[self.pctSelecc["pH"] == int(self.caolinPH.currentText())]
        # Ahora con el caolín de que tipo
        conCaolin = conPH[conPH["Blanco"] == (self.caolinTipo.currentText() == "Blanco")]
        print(conCaolin)
        # Generar las gráficas
        self.MplWidget.canvas.axes.clear()
        self.MplWidget.canvas.axes.plot(conCaolin["N"].tolist(), conCaolin["mu R"].tolist())
        self.MplWidget.canvas.axes.set_xlabel("Velocidad angular [r.p.m]")
        self.MplWidget.canvas.axes.set_ylabel("Viscosidad ( $\eta$ ) [mPA]")
        self.MplWidget.canvas.axes.set_position([0.18, 0.15, 0.8, 0.8])
        self.MplWidget.canvas.draw()
        
        self.MplWidget2.canvas.axes.clear()
        self.MplWidget2.canvas.axes.plot(np.log(np.array(conCaolin["gamma"].tolist())), np.log(np.array(conCaolin["tau"].tolist())))
        self.MplWidget2.canvas.axes.set_xlabel("ln(n)")
        self.MplWidget2.canvas.axes.set_ylabel("ln( $\tau$ )")
        self.MplWidget2.canvas.axes.set_position([0.18, 0.15, 0.8, 0.8])
        self.MplWidget2.canvas.draw()
        
        self.MplWidget3.canvas.axes.clear()
        self.MplWidget3.canvas.axes.plot(conCaolin["gamma"].tolist(), conCaolin["tau"].tolist())
        self.MplWidget3.canvas.axes.set_xlabel("Rápidez de deformación ( $\gamma$ ) [s$^-1$]")
        self.MplWidget3.canvas.axes.set_ylabel("Esfuerzo cortante ( $\tau$ ) [mPA]")
        self.MplWidget3.canvas.axes.set_position([0.18, 0.15, 0.8, 0.8])
        self.MplWidget3.canvas.draw()

        self.MplWidget4.canvas.axes.clear()
        self.MplWidget4.canvas.axes.plot(conCaolin["gamma"].tolist(), conCaolin["mu R"].tolist())
        self.MplWidget4.canvas.axes.set_xlabel("Rápidez de deformación ( $\gamma$ ) [s$^-1$]")
        self.MplWidget4.canvas.axes.set_ylabel("Viscosidad ( $\eta$ ) [mPA]")
        self.MplWidget4.canvas.axes.set_position([0.18, 0.15, 0.8, 0.8])
        self.MplWidget4.canvas.draw()
        
        print(self.MplWidget.canvas.axes.get_position)
    
    # Método para ajustar los datos
    def ajustarDatos(self):
        # Generar la lista de datos
        # Los porcentajes son los mandatorios con self.pctSelecc
        # Obtener la lista del color del Caolin
        conColor = self.pctSelecc[self.pctSelecc["Blanco"] == (self.caolinTipo.currentText() == "Blanco")]
        
        peHs = self.pctSelecc["pH"].unique().astype(int).tolist()
        peHs.reverse()
        
        # Conteo de Filas
        self.tableWidget.setRowCount( len(peHs)+1 )
        # Conteo de Columnas
        self.tableWidget.setColumnCount(5)
        self.tableWidget.setItem(0,0,QTableWidgetItem("pH"))
        self.tableWidget.setItem(0,1,QTableWidgetItem("mu"))
        self.tableWidget.setItem(0,2,QTableWidgetItem("R^2"))
        self.tableWidget.setItem(0,3,QTableWidgetItem("tau"))
        self.tableWidget.setItem(0,4,QTableWidgetItem("R^2"))
        
        i = 1
        # Recorrer los valores de pH para el caolín con Cw y color específicos
        for k in peHs:
            # Ahora obtener solo los datos del pH
            conPH = conColor[conColor["pH"] == k]
            # Crear el arreglo en x (rapidez deformación -gamma- )
            gamma = conPH["gamma"].tolist()
            # Crear el arreglo en y1 (viscosidad -mu-)
            mu = conPH["mu R"].tolist()
            # Crear el arreglo en y2 (esfuerzo cortante -tau-)
            tau = conPH["tau"].tolist()
            
            # Realizar el ajuste gamma - mu
            pars1, cov1 = curve_fit(f = potencias, xdata = gamma, ydata = mu)
            # Obtener la suma de residuos de los cuadrados
            r_cuad1 = obtenerRCuadrada(gamma, mu, pars1)
            pars2, cov2 = curve_fit(f = potencias, xdata = gamma, ydata = tau)
            # Obtener la suma de residuos de los cuadrados
            r_cuad2 = obtenerRCuadrada(gamma, tau, pars2)
            
            # Llenar la tabla
            self.tableWidget.setItem(i,0,QTableWidgetItem(str(k)))
            self.tableWidget.setItem(i,1,QTableWidgetItem(str(round(pars1[0],2))+" gamma^"+str(round(pars1[1],4))))
            self.tableWidget.setItem(i,2,QTableWidgetItem(str(round(r_cuad1,4))))
            self.tableWidget.setItem(i,3,QTableWidgetItem(str(round(pars2[0],4))+" gamma^"+str(round(pars2[1],4))))
            self.tableWidget.setItem(i,4,QTableWidgetItem(str(round(r_cuad2,4))))
            i += 1
        self.tableWidget.resizeColumnsToContents()
    
    # Modificar comportamiento tras dar check a autoajuste
    def cambiarAutoAjuste(self, state):
        if state == Qt.Checked:
            self.epok.setEnabled(False)
        else:
            self.epok.setEnabled(True)
    
    # Entrenar la red con los datos
    def entrenarRedDatos(self):
        # Generar la lista con los nombres de las columnas
        column_names = list(self.df.columns)
        
        # Copiar los datos en otra variable
        dataset = self.df.copy()
        
        # Datos de entrenamiento
        train_dataset = dataset.sample(frac=0.8,random_state=None)
        # Datos de prueba
        test_dataset = dataset.drop(train_dataset.index)
        
        # También se da una vista rápida a las estadísticas principales
        train_stats = train_dataset.describe()
        train_stats.pop("mu R")
        train_stats = train_stats.transpose()
        
        # Dividir las características de las etiquetas
        # Separar el valor objetivo, o "etiqueta", de las características. Esta etiqueta es el valor que será entrenado para predecir el modelo
        train_labels = train_dataset.pop('mu R')
        test_labels = test_dataset.pop('mu R')
        
        def norm(x): return(x - train_stats['mean']) / train_stats['std']
        
        # Normalizar tanto los datos de entrenamiento como de prueba
        normed_train_data = norm(train_dataset)
        normed_test_data = norm(test_dataset)
        
        # MODELO
        # Crear el modelo
        # Aquí se utiliza un modelo Sequential con dos capas ocultas densamente conectadas, y una capa de salida  que regresa un valor
        # continuo. Los pasos para la construcción del modelo están colocados en una función. 
        def build_model():
            # El modelo secuencial ingresa 9 puntos (los datos)
            # Una capa de 64 neuronas al medio
            # Salida de 1, el cual es el valor objetivo de la viscosidad
            model = keras.Sequential(
                    [layers.Dense(9, activation='relu', input_shape=[len(train_dataset.keys())]), 
                     layers.Dense(64, activation='relu'), 
                     layers.Dense(1)])
            # Un optimizador por medio de RMS con valor de 0.001
            optimizer = tf.keras.optimizers.RMSprop(0.001)
            # Se compila el modelo con la función de pérdida MSE, el optimizador y métricas MAE, MSE
            model.compile(loss='mse', optimizer=optimizer, metrics=['mae', 'mse'])
            return model
        
        # Generar el modelo
        model = build_model()
        
        # Mandar a consola el resumen del modelo
        model.summary()
        
        # ENTRENAR EL MODELO
        # Desplegar el proceso de entrenamiento al colocar un punto para cada época completada
        class PrintDot(keras.callbacks.Callback):
            def on_epoch_end(self, epoch, logs):
                if epoch % 100 == 0: print('')
                print('.', end='')
        
        # Obtener el valor de epocas a realizar dede el control
        EPOCHS = self.epok.value()
        
        # Si se coloca a entrenamiento por un máximo de época
        if self.autoEpok.isChecked() == False:
            # Historial de como se entrena el modelo
            history = model.fit(
              normed_train_data, train_labels,
              epochs=EPOCHS, validation_split = 0.2, verbose=0,
              callbacks=[PrintDot()])
        # o con un autodetener
        else:
            early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)
            history = model.fit(normed_train_data, train_labels, epochs=EPOCHS, validation_split = 0.2, verbose=0, callbacks=[early_stop, PrintDot()])
        
        # Visualizar el progreso del entrenamiento usando las estadísticas de entrenamiento del proceso usadas/almacenadas en el objeto history
        hist = pd.DataFrame(history.history)
        hist['epoch'] = history.epoch
        hist.tail()
        
        # UNDONE: Mandar un pequeño letrero de listo
        self.label_6.setVisible(True)
        self.label_6.setText("Listo")
        
        # Graficar
        hist = pd.DataFrame(history.history)
        hist['epoch'] = history.epoch
        self.MplWidget.canvas.axes.clear()
        self.MplWidget.canvas.axes.plot(hist['epoch'].tolist(), hist['mean_absolute_error'].tolist(), label='Entrenamiento')
        self.MplWidget.canvas.axes.plot(hist['epoch'].tolist(), hist['val_mean_absolute_error'].tolist(), label='Error de validación (MAE)')
        self.MplWidget.canvas.axes.set_xlabel("Epoca")
        self.MplWidget.canvas.axes.set_ylabel("Error Abs Prom (MAE) [$\mu$]")
        self.MplWidget.canvas.axes.set_position([0.18, 0.15, 0.8, 0.8])
        self.MplWidget.canvas.axes.legend()
        self.MplWidget.canvas.draw()
        
        self.MplWidget2.canvas.axes.clear()
        self.MplWidget2.canvas.axes.plot(hist['epoch'].tolist(), hist['mean_squared_error'].tolist(), label='Entrenamiento')
        self.MplWidget2.canvas.axes.plot(hist['epoch'].tolist(), hist['val_mean_squared_error'].tolist(), label='Error de validación (RMS)')
        self.MplWidget2.canvas.axes.set_xlabel("Epoca")
        self.MplWidget2.canvas.axes.set_ylabel("Error Cuadr. Medio (RMS) [$mu^2$]")
        self.MplWidget2.canvas.axes.set_position([0.20, 0.15, 0.7, 0.8])
        self.MplWidget2.canvas.axes.legend()
        self.MplWidget2.canvas.draw()
        
        # PREDICCIONES
        # Finalmente, realizar predicciones de los valores de viscosidad en el grupo de prueba
        test_predictions = model.predict(normed_test_data).flatten()
        
        # Distribución de los errores
        error = test_predictions - test_labels
        
        self.MplWidget3.canvas.axes.clear()
        self.MplWidget3.canvas.axes.scatter(test_labels, test_predictions, color="blue")
        self.MplWidget3.canvas.axes.plot([0, 1750], [0, 1750], color="red")
        self.MplWidget3.canvas.axes.set_xlabel("Valores reales \u03BC [mPa s]")
        self.MplWidget3.canvas.axes.set_ylabel("Predicciones \u03BC [mPa s]")
        self.MplWidget3.canvas.axes.set_position([0.18, 0.15, 0.8, 0.8])
        self.MplWidget3.canvas.draw()
        
        self.MplWidget4.canvas.axes.clear()
        self.MplWidget4.canvas.axes.hist(error, bins = 25)
        self.MplWidget4.canvas.axes.set_xlabel("Error de Predicción [\u03BC]")
        self.MplWidget4.canvas.axes.set_ylabel("Cuentas")
        self.MplWidget4.canvas.draw()
        
        # Salvar el modelo
        model.save(self.lineaArchivo_2.text())
        

# FUNCIÓN MAIN
if __name__ == "__main__":
    app =  QtWidgets.QApplication(sys.argv)
    window = MyApp()
    window.show()
    sys.exit(app.exec_())