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
        MainWindow.resize(515, 841)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.btn_load_video = QtWidgets.QPushButton(self.centralwidget)
        self.btn_load_video.setGeometry(QtCore.QRect(140, 30, 201, 31))
        self.btn_load_video.setObjectName("btn_load_video")
        self.btn_load_image = QtWidgets.QPushButton(self.centralwidget)
        self.btn_load_image.setGeometry(QtCore.QRect(140, 80, 201, 31))
        self.btn_load_image.setObjectName("btn_load_image")
        self.btn_load_folder = QtWidgets.QPushButton(self.centralwidget)
        self.btn_load_folder.setGeometry(QtCore.QRect(140, 130, 201, 31))
        self.btn_load_folder.setObjectName("btn_load_folder")
        self.groupBox = QtWidgets.QGroupBox(self.centralwidget)
        self.groupBox.setGeometry(QtCore.QRect(60, 210, 371, 91))
        self.groupBox.setObjectName("groupBox")
        self.btn_background_subtraction = QtWidgets.QPushButton(self.groupBox)
        self.btn_background_subtraction.setGeometry(QtCore.QRect(80, 30, 201, 31))
        self.btn_background_subtraction.setObjectName("btn_background_subtraction")
        self.groupBox_2 = QtWidgets.QGroupBox(self.centralwidget)
        self.groupBox_2.setGeometry(QtCore.QRect(60, 470, 371, 91))
        self.groupBox_2.setObjectName("groupBox_2")
        self.btn_perspective_transform = QtWidgets.QPushButton(self.groupBox_2)
        self.btn_perspective_transform.setGeometry(QtCore.QRect(80, 30, 201, 31))
        self.btn_perspective_transform.setObjectName("btn_perspective_transform")
        self.groupBox_3 = QtWidgets.QGroupBox(self.centralwidget)
        self.groupBox_3.setGeometry(QtCore.QRect(60, 580, 371, 131))
        self.groupBox_3.setObjectName("groupBox_3")
        self.btn_image_reconstruction = QtWidgets.QPushButton(self.groupBox_3)
        self.btn_image_reconstruction.setGeometry(QtCore.QRect(80, 30, 201, 31))
        self.btn_image_reconstruction.setObjectName("btn_image_reconstruction")
        self.btn_compute_the_recostruction_error = QtWidgets.QPushButton(self.groupBox_3)
        self.btn_compute_the_recostruction_error.setGeometry(QtCore.QRect(80, 70, 201, 31))
        self.btn_compute_the_recostruction_error.setObjectName("btn_compute_the_recostruction_error")
        self.groupBox_4 = QtWidgets.QGroupBox(self.centralwidget)
        self.groupBox_4.setGeometry(QtCore.QRect(60, 320, 371, 131))
        self.groupBox_4.setObjectName("groupBox_4")
        self.btn_preprocessing = QtWidgets.QPushButton(self.groupBox_4)
        self.btn_preprocessing.setGeometry(QtCore.QRect(80, 30, 201, 31))
        self.btn_preprocessing.setObjectName("btn_preprocessing")
        self.btn_video_tracking = QtWidgets.QPushButton(self.groupBox_4)
        self.btn_video_tracking.setGeometry(QtCore.QRect(80, 70, 201, 31))
        self.btn_video_tracking.setObjectName("btn_video_tracking")
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 515, 21))
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
        self.btn_load_video.setText(_translate("MainWindow", "Load Video"))
        self.btn_load_image.setText(_translate("MainWindow", "Load Image"))
        self.btn_load_folder.setText(_translate("MainWindow", "Load Folder"))
        self.groupBox.setTitle(_translate("MainWindow", "1. Background Subtraction"))
        self.btn_background_subtraction.setText(_translate("MainWindow", "1.1Background Subtraction"))
        self.groupBox_2.setTitle(_translate("MainWindow", "3.Perspective Transform"))
        self.btn_perspective_transform.setText(_translate("MainWindow", "3.1Perspective Transform"))
        self.groupBox_3.setTitle(_translate("MainWindow", "4.PCA"))
        self.btn_image_reconstruction.setText(_translate("MainWindow", "4.1Image Reconstruction"))
        self.btn_compute_the_recostruction_error.setText(_translate("MainWindow", "4.2Compute the Reconstruction Error"))
        self.groupBox_4.setTitle(_translate("MainWindow", "2. Optical Flow"))
        self.btn_preprocessing.setText(_translate("MainWindow", "2.1Preprocessing"))
        self.btn_video_tracking.setText(_translate("MainWindow", "2.2Video Tracking"))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())

