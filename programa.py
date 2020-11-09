# -*- coding: utf-8 -*-
import sys

# # #Importar aquí las librerías a utilizar # # #
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

# Librerías del PyQt
from PyQt5 import uic, QtWidgets, QtCore, QtGui
from PyQt5.QtWidgets import QApplication, QWidget, QInputDialog, QLineEdit, QFileDialog, QTableWidget, QTableWidgetItem, QVBoxLayout
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
        
        # Conectar acciones de Widgets con funciones de la clase
        self.botonAbrir.clicked.connect(self.openFileNameDialog)
        self.caolinPct.currentTextChanged.connect(self.selectPct)
        self.caolinPH.currentTextChanged.connect(self.selectPH)
        #self.caolinTipo.currentTextChanged.connect(self.final)
        self.graficar.clicked.connect(self.graficarDatos)
        self.ajustar.clicked.connect(self.ajustarDatos)
                        
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
            # Abrir mediante pandas el archivo CSV
            self.df = pd.read_csv(fileName)
            # Cargar los datos en el primer combobox
            # TODO: Modificar para cambiar a porcentaje
            pcts = [i * 100 for i in self.df["Cw"].unique().tolist()]
            print(pcts)
            
            # Cargar los datos en el combobox
            self.caolinPct.addItems(self.df["Cw"].unique().astype(str).tolist())
            # Activar el combobox del porcentaje
            self.caolinPct.setEnabled(True)
    
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

# FUNCIÓN MAIN
if __name__ == "__main__":
    app =  QtWidgets.QApplication(sys.argv)
    window = MyApp()
    window.show()
    sys.exit(app.exec_())