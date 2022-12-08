from PyQt5 import QtCore, QtWidgets
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QFileDialog
import tensorflow as tf
import cv2 , numpy as np
import keras
from keras.models import load_model
import pickle
import keras
import os
from Resnet import Resnet50
import matplotlib.pyplot as plt
from UI import Ui_MainWindow
from keras.preprocessing.image import load_img, img_to_array
from keras.applications.vgg16 import preprocess_input

IMAGE_SIZE = (224, 224)
class MainWindow_controller(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__() # in python3, super(Class, self).xxx = super().xxx
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.setup_control()


    def setup_control(self):
        self.ui.btn_load_image.clicked.connect(self.load_image)
        self.ui.btn_show_images.clicked.connect(self.show_images)
        self.ui.btn_show_distribution.clicked.connect(self.distribution)
        self.ui.btn_show_model_structure.clicked.connect(self.model_structure)
        # self.ui.btn_show_comparison.clicked.connect(self.accuracy_loss)
        # self.ui.btn_inference.clicked.connect(self.inference)



    def load_image(self):
        img_path, _ = QFileDialog.getOpenFileName(
            self,
            filter='Image Files (*.png *.jpg *.jpeg *.bmp)')           # start path
        print(img_path)

        self.img = cv2.imread("{}".format(img_path))
        self.img = cv2.resize(self.img, IMAGE_SIZE, interpolation=cv2.INTER_CUBIC)
        height, width, channel = self.img.shape
        bytesPerline = 3 * width
        qimg = QImage(self.img.tobytes(), width, height, bytesPerline, QImage.Format_RGB888).rgbSwapped()
        qpixmap = QPixmap.fromImage(qimg)
        qpixmap_height = qpixmap.height()
        qpixmap_height += 100
        scaled_pixmap = qpixmap.scaledToHeight(qpixmap_height)
        self.ui.label.setPixmap(scaled_pixmap)


    def show_images(self):

        self.class_list = ["cat", "dog"]
        title_list = []
        image_list = []
        fig = plt.figure(figsize=(1, 2))
        fig.subplots_adjust(hspace=0.1, wspace=0.1)
        cat = cv2.imread("inference_dataset/Cat/8043.jpg")
        dog = cv2.imread("inference_dataset/Dog/12053.jpg")
        cat = cv2.resize(cat, IMAGE_SIZE, interpolation=cv2.INTER_CUBIC)
        dog = cv2.resize(dog, IMAGE_SIZE, interpolation=cv2.INTER_CUBIC)
        cat = cat[:,:,::-1]
        dog = dog[:,:,::-1]
        image_list.append(cat)
        image_list.append(dog)
        for i in range(1,3):
            title_list.append(self.class_list[i-1])
            plt.subplot(120 + i )
            plt.imshow(image_list[i-1])
            plt.title(title_list[i-1])
            plt.axis('off')
        plt.show()
    
    def distribution(self):
        # trcat = 'C:/Users/user/Desktop/cvdl_HW2/HW2_05/Dataset_OpenCvDl_Hw2_Q5/training_dataset/Cat/'
        # trdog = 'C:/Users/user/Desktop/cvdl_HW2/HW2_05/Dataset_OpenCvDl_Hw2_Q5/training_dataset/Dog/'
        # valcat = 'C:/Users/user/Desktop/cvdl_HW2/HW2_05/Dataset_OpenCvDl_Hw2_Q5/validation_dataset/Cat/'
        # valdog = 'C:/Users/user/Desktop/cvdl_HW2/HW2_05/Dataset_OpenCvDl_Hw2_Q5/validation_dataset/Dog/'
        # num = [len(os.listdir(trcat))+len(os.listdir(valcat)),len(os.listdir(trdog))+len(os.listdir(valdog))]
        # x_label = ["cat", "dog"]
        # plt.xticks( range(2),x_label)  
        # plt.bar(range(2),num, align = 'center',color= 'steelblue', alpha = 0.8)
        # plt.ylabel('number of images')
        # plt.title('Class Distribution')
        # for x,y in enumerate(num):plt.text(x,y,'%s'%y,ha='center')
        # plt.show()

        img_src = cv2.imread('distribution.png')
        cv2.imshow("result", img_src)

    
    def model_structure(self):
        model = Resnet50().model()
        print(model.summary())

