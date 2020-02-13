import sys, time
import numpy as np
import pandas as pd
from CliffordSpace import Cl
from CliffordNumbers import ClNumber, ClVector

from matplotlib.backends.qt_compat import QtCore, QtWidgets, QtGui
from matplotlib.backends.backend_qt5agg import (FigureCanvas, NavigationToolbar2QT as NavigationToolbar)
from matplotlib.figure import Figure

class ApplicationWindow(QtWidgets.QMainWindow):
    
    def __init__(self):
        
        super().__init__()
        self._main = QtWidgets.QWidget()
        self.setCentralWidget(self._main)
        self.vLayout = QtWidgets.QVBoxLayout(self._main)

        #------------------- WINDOW SETTINGS -------------------#
        #self.setGeometry(300,300,1000,600)
        self.setWindowTitle("Deep Features GUI")
        #self.setWindowIcon(QtGui.QIcon(""))
        self.setStyleSheet("background-color:")
        self.show()

        #------------------- WINDOW SETTINGS -------------------#
        self.fs=100
        self.num=2000

        #------------------- DEFINE FEATURE SPACE -------------------#
        self.boxValues, self.weightNames = list(), list()

        hLayout = QtWidgets.QHBoxLayout()
        self.loadPath = QtWidgets.QTextEdit(maximumHeight=30)
        hLayout.addWidget(self.loadPath)
        self.loadFile = QtWidgets.QPushButton('Load File', self)
        self.loadFile.clicked.connect(self.LoadFile)
        hLayout.addWidget(self.loadFile)
        self.vLayout.addLayout(hLayout)

        hLayout = QtWidgets.QHBoxLayout()
        numClasses = QtWidgets.QLabel('Number of Classes:')
        hLayout.addWidget(numClasses)
        self.classes = QtWidgets.QSpinBox(minimum=2,maximum=10, value=3)
        hLayout.addWidget(self.classes)
        self.vLayout.addLayout(hLayout)
        hLayout = QtWidgets.QHBoxLayout()
        numDimensions = QtWidgets.QLabel('Number of Dimensions:')
        hLayout.addWidget(numDimensions)
        self.dimensions = QtWidgets.QSpinBox(minimum=2,maximum=128, value=5)
        hLayout.addWidget(self.dimensions)
        self.vLayout.addLayout(hLayout)

        hLayout = QtWidgets.QHBoxLayout()
        self.space = QtWidgets.QPushButton('Create Space')
        hLayout.addWidget(self.space)
        self.space.clicked.connect(self.CreateSpace)
        self.normAll = QtWidgets.QPushButton('Normalize')
        hLayout.addWidget(self.normAll)
        self.normAll.clicked.connect(self.Normalize)
        self.calcul = QtWidgets.QPushButton('Calculate')
        hLayout.addWidget(self.calcul)
        self.calcul.clicked.connect(self.Calculate)
        self.progress = QtWidgets.QProgressBar(self, maximumWidth=200)
        hLayout.addWidget(self.progress)
        self.vLayout.addLayout(hLayout)

        self.static_canvas = FigureCanvas(Figure(figsize=(5,3)))
        self.toolbar = NavigationToolbar(self.static_canvas, self)
        self.toPlot = {
            '00': [np.arange(20), np.zeros((1,20)),'Norm' ],
            '01': [np.arange(20), np.zeros((1,20)),'Angle'],
            '10': [np.arange(20), np.zeros((1,20)),'Norm Derivative' ],
            '11': [np.arange(20), np.zeros((1,20)),'Angle Derivative']
                       }
        self.Plot()
        self.CreateSpace()

    def LoadFile(self):
        
        self.ClearSpace()
        name,_ = QtWidgets.QFileDialog.getOpenFileName(self, 'Open File','./','All files(*)')
        df = pd.read_csv(name)
        self.weights = list()
        counter = 0
        for name, value in df.items():
            if counter==0:
                self.a = np.array(value)
            else:
                self.weights.append(np.array(value))
            counter += 1
        self.weights = np.array(self.weights)
        self.classes.setValue(len(self.weights))
        self.dimensions.setValue(len(self.weights[0]))
        for i in range(self.classes.value()+1):
            weightValues = list()
            hLayout = QtWidgets.QHBoxLayout()
            if i==0:
                weightName = QtWidgets.QLabel('Ae:')
            else:
                weightName = QtWidgets.QLabel('W'+str(i)+':')
            hLayout.addWidget(weightName)
            for j in range(self.dimensions.value()):
                if i==0:
                    weightValues.append(QtWidgets.QDoubleSpinBox(minimum=-999.999, maximum=999.999, maximumWidth=81,singleStep=0.1, value=self.a[j], decimals=3))
                    hLayout.addWidget(weightValues[j])
                else:
                    weightValues.append(QtWidgets.QDoubleSpinBox(minimum=-999.999, maximum=999.999, maximumWidth=81,singleStep=0.1, value=self.weights[i-1,j], decimals=3))
                    hLayout.addWidget(weightValues[j])

            self.vLayout.addLayout(hLayout)
            self.boxValues.append(weightValues)
            self.weightNames.append(weightName)
        self.boxValues = np.array(self.boxValues)
        
        pass

        
    def Plot(self):

        self.static_canvas.setParent(None)
        self.removeToolBar(self.toolbar)
        self.static_canvas = FigureCanvas(Figure(figsize=(5,3)))
        self.toolbar = NavigationToolbar(self.static_canvas, self)
        self.addToolBar(self.toolbar)
        self._static_ax = self.static_canvas.figure.subplots(2,2)

        for key, value in self.toPlot.items():
            self._static_ax[int(key[0]),int(key[1])].grid(True)
            self._static_ax[int(key[0]),int(key[1])].set_title(value[2])
            for i in range(len(value[1])):
                self._static_ax[int(key[0]),int(key[1])].plot(value[0],value[1][i],label='Class '+str(i+1))
            self._static_ax[int(key[0]),int(key[1])].legend()
        
        self.vLayout.addWidget(self.static_canvas)
        pass

    def CreateSpace(self):
        
        self.ClearSpace()
        for i in range(self.classes.value()+1):
            weightValues = list()
            hLayout = QtWidgets.QHBoxLayout()
            if i==0:
                weightName = QtWidgets.QLabel('Ae:')
            else:
                weightName = QtWidgets.QLabel('W'+str(i)+':')
            hLayout.addWidget(weightName)
            for j in range(self.dimensions.value()):
                weightValues.append(QtWidgets.QDoubleSpinBox(minimum=-999.999, maximum=999.999, maximumWidth=81,singleStep=0.1, value=np.random.randn(1), decimals=3))
                hLayout.addWidget(weightValues[j])
            self.vLayout.addLayout(hLayout)
            self.boxValues.append(weightValues)
            self.weightNames.append(weightName)
        self.boxValues = np.array(self.boxValues)
        self.a = np.array([self.boxValues[0,i].value() for i in range(len(self.boxValues[0]))])
        self.weights = np.array([[self.boxValues[i,j].value() for j in range(len(self.boxValues[i]))] for i in range(1,len(self.boxValues))])
        pass

    def RecreateSpace(self):
        
        self.ClearSpace(False)
        for i in range(len(self.weightNames)):
            hLayout = QtWidgets.QHBoxLayout()
            hLayout.addWidget(self.weightNames[i])
            for j in range(len(self.weights[0])):
                hLayout.addWidget(self.boxValues[i,j])
            self.vLayout.addLayout(hLayout)
        pass

    def ClearSpace(self,fully=True):

        [[self.boxValues[i][j].setParent(None) for j in range(len(self.boxValues[0]))] for i in range(len(self.boxValues))]
        [self.weightNames[i].setParent(None) for i in range(len(self.weightNames))]
        if fully:
            self.boxValues=list()
            self.weightNames=list()
        pass

    def Normalize(self):

        self.a = self.a/np.linalg.norm(self.a)
        [self.boxValues[0,i].setValue(self.a[i]) for i in range(len(self.a))]
        #norm = np.linalg.norm(self.weights,axis=1)
        #self.weights =  np.array([np.divide(self.weights[i],norm[i]) for i in range(len(norm))])
        #[[self.boxValues[i+1,j].setValue(self.weights[i,j]) for j in range(len(self.weights[i]))] for i in range(len(self.weights))]
        pass

    def Calculate(self):

        self.Scale()
        self.progress.setValue(25)
        self.ScaleDerivative()
        self.progress.setValue(50)
        self.Rotate()
        self.progress.setValue(75)
        self.RotateDerivative()
        self.progress.setValue(99)
        self.Plot()
        self.RecreateSpace()
        self.progress.setValue(100)
        pass

    def Softmax(self,x):

        x = np.array(x)
        return np.exp(x) / sum(np.exp(x))

    def Scale(self):

        norms = np.arange(self.num)/self.fs
        self.ind = self.planeOfRotation()
        rOutputs = list()
        
        for i in range(len(self.weights)):
            
            w_Cl = ClVector(Cl(len(self.weights[i])),self.weights[i])
            proj_w = self.rotateNd(w_Cl**self.pOR,-self.pOR,np.pi/2)._transform2numpy()
            rOutputs.append(norms*np.linalg.norm(self.a)*np.linalg.norm(proj_w)*np.cos(self.angleBetVectors(proj_w,self.a)))
        
        rOutputs = np.array(rOutputs)
        rOutputs = np.array([self.Softmax(rOutputs[:,i]) for i in range(len(rOutputs[0]))]) 
        
        #---------------------- PLOTTING ----------------------#
        self.rOutputs = np.transpose(rOutputs)
        self.toPlot['00'] = [norms, np.array([self.rOutputs[i] for i in range(len(self.rOutputs))]), 'Norm']
        pass

    def ScaleDerivative(self):

        norms = np.arange(self.num)/self.fs
        #self.planeOfRotation()
        rS, rDS = list(), list()
        
        for i in range(len(self.weights)):
            
            w_Cl = ClVector(Cl(len(self.weights[i])),self.weights[i])
            proj_w = self.rotateNd(w_Cl**self.pOR,-self.pOR,np.pi/2)._transform2numpy()
            rS.append(norms*np.linalg.norm(self.a)*np.linalg.norm(proj_w)*np.cos(self.angleBetVectors(proj_w,self.a)))
            rDS.append(np.ones(self.num)*np.linalg.norm(self.a)*np.linalg.norm(proj_w)*np.cos(self.angleBetVectors(proj_w,self.a)))
        
        rDerivatives = np.array([self.DSDz(np.array(rS),np.array(rDS),i) for i in range(self.classes.value())])
        
        #---------------------- PLOTTING ----------------------#
        self.rDerivatives = np.array(rDerivatives)
        self.toPlot['10'] = [norms, self.rDerivatives, 'Norm Derivative']
        pass

    def Rotate(self):

        thetas = np.arange(self.num)/(10*self.fs)*np.pi
        e = 0.0001
        #ind = self.planeOfRotation()
        w_Cl = ClVector(Cl(len(self.weights[self.ind])),self.weights[self.ind])
        proj_wj =  self.rotateNd(w_Cl**self.pOR,-self.pOR,np.pi/2)._transform2numpy()
        wja = self.angleBetVectors(proj_wj, self.a)
        aOutputs = list()

        for i in range(len(self.weights)):
            
            w_Cl = ClVector(Cl(len(self.weights[i])),self.weights[i])
            proj_w = self.rotateNd(w_Cl**self.pOR,-self.pOR,np.pi/2)._transform2numpy()
            wia = self.angleBetVectors(proj_w, self.a)
            wij = self.angleBetVectors(proj_w, proj_wj)
            if abs(wia+wij-wja)<e or abs(wia-wij-wja)<e:
                wia = -wia
            aOutputs.append(np.linalg.norm(self.a)*np.linalg.norm(proj_w)*np.cos(thetas+wia))

        aOutputs = np.array(aOutputs)
        aOutputs = np.array([self.Softmax(aOutputs[:,i]) for i in range(len(aOutputs[0]))]) 

        #---------------------- PLOTTING ----------------------#
        self.aOutputs = np.transpose(aOutputs)
        self.toPlot['01'] = [thetas, np.array([self.aOutputs[i] for i in range(len(self.aOutputs))]), 'Angle']
        pass

    def RotateDerivative(self):

        thetas = np.arange(self.num)/(10*self.fs)*np.pi
        e = 0.0001
        #ind = self.planeOfRotation()
        w_Cl = ClVector(Cl(len(self.weights[self.ind])),self.weights[self.ind])
        proj_wj =  self.rotateNd(w_Cl**self.pOR,-self.pOR,np.pi/2)._transform2numpy()
        wja = self.angleBetVectors(proj_wj, self.a)
        aS, aDS = list(), list()

        for i in range(len(self.weights)):
            
            w_Cl = ClVector(Cl(len(self.weights[i])),self.weights[i])
            proj_w = self.rotateNd(w_Cl**self.pOR,-self.pOR,np.pi/2)._transform2numpy()
            wia = self.angleBetVectors(proj_w, self.a)
            wij = self.angleBetVectors(proj_w, proj_wj)
            if abs(wia+wij-wja)<e or abs(wia+wij-wja)<e:
                wia = -wia
            aS.append(np.linalg.norm(self.a)*np.linalg.norm(proj_w)*np.cos(thetas+wia))
            aDS.append(-1*np.linalg.norm(self.a)*np.linalg.norm(proj_w)*np.sin(thetas+wia))

        aDerivatives = np.array([self.DSDz(np.array(aS),np.array(aDS),i) for i in range(self.classes.value())])

        #---------------------- PLOTTING ----------------------#
        self.aDerivatives = np.array(aDerivatives)
        self.toPlot['11'] = [thetas, self.aDerivatives, 'Angle Derivative']
        pass

    def angleBetVectors(self, vector1, vector2):
        arc = np.dot(vector1,vector2)/(np.linalg.norm(vector1)*np.linalg.norm(vector2))
        if arc > 1 and arc < 1.0001:
            arc = 1
        return np.arccos(arc)

    def planeOfRotation(self):
        
        ind = np.argmin([self.angleBetVectors(self.a, self.weights[i]) for i in range(len(self.weights))])
        a_Cl = ClVector(Cl(len(self.a)), self.a)
        w_Cl = ClVector(Cl(len(self.weights[ind])), self.weights[ind])
        self.pOR = (a_Cl^w_Cl)._normalize()
        return ind

    def rotateNd(self, nVector, rotationPlane, rotationTheta=0):
    
        cl = Cl(nVector.dimensions)
        rotor  = ClNumber(cl,{'': np.cos(rotationTheta/2)}) - np.sin(rotationTheta/2)*rotationPlane
        rotorS = ClNumber(cl,{'': np.cos(rotationTheta/2)}) + np.sin(rotationTheta/2)*rotationPlane
        return rotor*nVector*rotorS

    def DSDz(self,sigs,dSigs,ind):
        
        numerator = np.sum([(dSigs[ind]-dSigs[c])*np.exp(s) for c,s in enumerate(sigs)],axis=0)
        denominator = np.sum([np.exp(s) for s in sigs],axis=0)
        coef = self.Softmax(sigs)[ind]
        return coef*numerator/denominator


if __name__ == "__main__":
    qapp = QtWidgets.QApplication(sys.argv)
    app = ApplicationWindow()
    app.show()
    qapp.exec_()