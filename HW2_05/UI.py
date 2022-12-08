# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'UI.ui'
#
# Created by: PyQt5 UI code generator 5.9.2
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(800, 599)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.groupBox = QtWidgets.QGroupBox(self.centralwidget)
        self.groupBox.setGeometry(QtCore.QRect(60, 90, 191, 391))
        self.groupBox.setObjectName("groupBox")
        self.btn_load_image = QtWidgets.QPushButton(self.groupBox)
        self.btn_load_image.setGeometry(QtCore.QRect(50, 70, 75, 23))
        self.btn_load_image.setObjectName("btn_load_image")
        self.btn_show_images = QtWidgets.QPushButton(self.groupBox)
        self.btn_show_images.setGeometry(QtCore.QRect(10, 130, 161, 23))
        self.btn_show_images.setObjectName("btn_show_images")
        self.btn_show_model_structure = QtWidgets.QPushButton(self.groupBox)
        self.btn_show_model_structure.setGeometry(QtCore.QRect(10, 210, 161, 23))
        self.btn_show_model_structure.setObjectName("btn_show_model_structure")
        self.btn_show_distribution = QtWidgets.QPushButton(self.groupBox)
        self.btn_show_distribution.setGeometry(QtCore.QRect(10, 170, 161, 23))
        self.btn_show_distribution.setObjectName("btn_show_distribution")
        self.btn_show_comparison = QtWidgets.QPushButton(self.groupBox)
        self.btn_show_comparison.setGeometry(QtCore.QRect(10, 250, 161, 23))
        self.btn_show_comparison.setObjectName("btn_show_comparison")
        self.btn_inference = QtWidgets.QPushButton(self.groupBox)
        self.btn_inference.setGeometry(QtCore.QRect(10, 290, 161, 23))
        self.btn_inference.setObjectName("btn_inference")
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(310, 80, 361, 361))
        self.label.setText("")
        self.label.setObjectName("label")
        self.label_2 = QtWidgets.QLabel(self.centralwidget)
        self.label_2.setGeometry(QtCore.QRect(296, 12, 311, 71))
        self.label_2.setObjectName("label_2")
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 800, 21))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.groupBox.setTitle(_translate("MainWindow", "5.Resnet10 Test"))
        self.btn_load_image.setText(_translate("MainWindow", "Load Image"))
        self.btn_show_images.setText(_translate("MainWindow", "1.Show  Images"))
        self.btn_show_model_structure.setText(_translate("MainWindow", "3.Show Model Structure"))
        self.btn_show_distribution.setText(_translate("MainWindow", "2.Show Distribution"))
        self.btn_show_comparison.setText(_translate("MainWindow", "4.Show Comparison"))
        self.btn_inference.setText(_translate("MainWindow", "5.Inference"))
        self.label_2.setText(_translate("MainWindow", "TextLabel"))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())

