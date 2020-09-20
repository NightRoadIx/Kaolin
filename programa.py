# -*- coding: utf-8 -*-
import sys

# # #Importar aquí las librerías a utilizar # # #
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Librerías del PyQt
from PyQt5 import uic, QtWidgets, QtCore, QtGui
from PyQt5.QtWidgets import QApplication, QWidget, QInputDialog, QLineEdit, QFileDialog, QTableWidget, QTableWidgetItem, QVBoxLayout
from PyQt5.QtGui import QIcon
from PyQt5.QtCore import Qt, pyqtSlot

from matplotlib.backends.backend_qt5agg import (NavigationToolbar2QT as NavigationToolbar)

# Nombre del archivo
qtCreatorFile = "ventagui.ui"

Ui_MainWindow, QtBaseClass = uic.loadUiType(qtCreatorFile)

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
        
        # Conectar acciones de Widgets con funciones de la clase
        self.botonAbrir.clicked.connect(self.openFileNameDialog)
        self.caolinPct.currentTextChanged.connect(self.selectPct)
        self.caolinPH.currentTextChanged.connect(self.selectPH)
        #self.caolinTipo.currentTextChanged.connect(self.final)
        self.graficar.clicked.connect(self.graficarDatos)
                        
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
        
        # CrearTabla
        #self.crearTabla() 
            
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
        self.MplWidget.canvas.axes.set_ylabel("Viscosidad ($\eta$) [mPA]")
        self.MplWidget.canvas.draw()
        
        self.MplWidget2.canvas.axes.clear()
        self.MplWidget2.canvas.axes.plot(np.log(np.array(conCaolin["gamma"].tolist())), np.log(np.array(conCaolin["tau"].tolist())))
        self.MplWidget2.canvas.axes.set_xlabel("ln(n)")
        self.MplWidget2.canvas.axes.set_ylabel("ln($\tau$)")
        self.MplWidget2.canvas.draw()
        
        self.MplWidget3.canvas.axes.clear()
        self.MplWidget3.canvas.axes.plot(conCaolin["gamma"].tolist(), conCaolin["tau"].tolist())
        self.MplWidget3.canvas.axes.set_xlabel("Rápidez de deformación ($\gamma$) [s$^-1$]")
        self.MplWidget3.canvas.axes.set_ylabel("Esfuerzo cortante ($\tau$) [mPA]")
        self.MplWidget3.canvas.draw()

        self.MplWidget4.canvas.axes.clear()
        self.MplWidget4.canvas.axes.plot(conCaolin["gamma"].tolist(), conCaolin["mu R"].tolist())
        self.MplWidget4.canvas.axes.set_xlabel("Rápidez de deformación ($\gamma$) [s$^-1$]")
        self.MplWidget4.canvas.axes.set_ylabel("Viscosidad ($\eta$) [mPA]")
        self.MplWidget4.canvas.draw()
    
    '''def crearTabla(self):
        # primero se obtiene la lista de pH
        uno = self.df[self.df["Cw"] == float(self.caolinPct.currentText())]
        #dos = uno[uno["pH"] == self.caolinPH.currentText() ]
        print( type(uno["pH"].tolist()[0]) )
        print("------")
        #conPH = self.pctSelecc[self.pctSelecc["pH"] == int(self.caolinPH.currentText())]
        # Ahora con el caolín de que tipo
        conCaolin = conPH[conPH["Blanco"] == (self.caolinTipo.currentText() == "Blanco")]
        
        # Generar el número de filas
        self.tableWidget.setRowCount(conCaolin.shape[0]+1)
  
        # Generar el número de columnas
        self.tableWidget.setColumnCount(conCaolin.shape[1])
        
        i, j = 0, 0
        # Colocar los nombres de las columnas
        for columnas in conCaolin.columns:
            self.tableWidget.setItem(i,j, QTableWidgetItem(str(columnas)))
            j += 1
        # Subir los datos
        for fila in range(conCaolin.shape[0]):
            i += 1
            j = 0
            for columna in range(conCaolin.shape[1]):
                print(conCaolin.iloc[fila].tolist()[columna])
                self.tableWidget.setItem(i,j, QTableWidgetItem(str(conCaolin.iloc[fila].tolist()[columna])))
                j += 1'''

# FUNCIÓN MAIN
if __name__ == "__main__":
    app =  QtWidgets.QApplication(sys.argv)
    window = MyApp()
    window.show()
    sys.exit(app.exec_())