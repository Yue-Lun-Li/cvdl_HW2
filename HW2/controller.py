from PyQt5 import QtCore, QtWidgets
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QFileDialog
import numpy as np
import cv2 
import cv2.aruco as aruco
import matplotlib.pyplot as plt
from  matplotlib.pyplot import imshow
from sklearn.decomposition import PCA
from UI import Ui_MainWindow



class MainWindow_controller(QtWidgets.QMainWindow):

    params = cv2.SimpleBlobDetector_Params()

    def __init__(self):
        super().__init__() # in python3, super(Class, self).xxx = super().xxx
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.setup_control()

        self.params.minThreshold = 10
        self.params.maxThreshold = 200
        self.params.filterByArea = True
        self.params.minArea = 35
        self.params.filterByCircularity = True
        self.params.minCircularity = 0.8
        self.params.maxCircularity = 0.9
        self.params.filterByConvexity = True
        self.params.minConvexity = 0.5
        self.params.filterByInertia = True
        self.params.minInertiaRatio = 0.5



    def setup_control(self):
        self.ui.btn_load_folder.clicked.connect(self.load_folder)
        self.ui.btn_load_image.clicked.connect(self.load_image)
        self.ui.btn_load_video.clicked.connect(self.load_video)
        self.ui.btn_background_subtraction.clicked.connect(self.background_subtraction)
        self.ui.btn_preprocessing.clicked.connect(self.preprocessing)
        self.ui.btn_video_tracking.clicked.connect(self.video_tracking)
        self.ui.btn_perspective_transform.clicked.connect(self.perspective_transform)
        self.ui.btn_image_reconstruction.clicked.connect(self.image_reconstruction)
        self.ui.btn_compute_the_recostruction_error.clicked.connect(self.reconstruct_err)

    def load_folder(self):
        self.folder_path = QFileDialog.getExistingDirectory(self,
                  "Open folder",
                  "./")                 # start path
        print(self.folder_path)

    def load_image(self):
        self.img, _ = QFileDialog.getOpenFileName(
            self,
            filter='Image Files (*.png *.jpg *.jpeg *.bmp)')           # start path
        print(self.img)

    def load_video(self):
        self.vdieo, _ = QFileDialog.getOpenFileName(
            self,
            filter='Image Files (*.mp4)')           # start path
        print(self.vdieo)   

    def background_subtraction (self):
        i = mean = std = 0
        frames = []
        cap = cv2.VideoCapture("{}".format(self.vdieo) )
        fps = cap.get(cv2.CAP_PROP_FPS)


        
        while (cap.isOpened()):
            ret , frame = cap.read()
            if not ret:
                break
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            mask = np.zeros_like(gray)

            if i < 25 :
                frames.append(gray)
            elif i == 25:
                frames.append(gray)
                frames = np.array(frames)
                mean = np.mean(frames, axis=0)
                std = np.std(frames, axis=0)
                std[std < 5] = 5
            else:
                diff = np.subtract(gray, mean)
                diff = np.absolute(diff)
                mask[diff > 5*std] = 255

            result = cv2.bitwise_and(frame, frame, mask = mask)

            mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

            result = np.hstack((frame, mask, result))
            
            cv2.imshow('1.1 Background Subtraction', result)

            if cv2.waitKey(int(1000/fps)) & 0xFF == ord('q'):
                break

            i += 1

        cap.release()
        cv2.destroyAllWindows()

    def preprocessing(self):

        detector = cv2.SimpleBlobDetector_create(self.params)

        cap = cv2.VideoCapture("{}".format(self.vdieo) )
        ret, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        keypoints = detector.detect(gray)

        for kp in keypoints:
            x, y = map(lambda x: int(x), kp.pt)
            
            frame = cv2.rectangle(frame, (x - 6, y - 6), (x + 6, y + 6), (0, 0, 255), 1)
            frame = cv2.line(frame, (x, y - 6), (x, y + 6), (0, 0, 255), 1)
            frame = cv2.line(frame, (x - 6, y), (x + 6, y), (0, 0, 255), 1)

        cv2.imshow('2.1 Preprocessing', frame)
        cv2.waitKey(0)
        cap.release()
        cv2.destroyAllWindows()

    def video_tracking(self):
        lk_params = dict(
            winSize = (15,15),
            maxLevel = 2,
            criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

        detector = cv2.SimpleBlobDetector_create(self.params)


        capture = cv2.VideoCapture("{}".format(self.vdieo) )
        fps = capture.get(cv2.CAP_PROP_FPS)

        ret, frame = capture.read()
        gray_1 = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        keypoints = detector.detect(gray_1)

        p0 = np.array([[[kp.pt[0], kp.pt[1]]] for kp in keypoints]).astype(np.float32)
        mask = np.zeros_like(frame)

        while(capture.isOpened()):

            ret, frame = capture.read()
            if not ret:
                break
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            p1, st, _ = cv2.calcOpticalFlowPyrLK(gray_1, gray, p0, None, **lk_params)

            good_new = p1[st==1]
            good_old = p0[st==1]


            for i,(new,old) in enumerate(zip(good_new, good_old)):
                a, b = new.ravel()
                c, d = old.ravel()

                mask = cv2.line(mask, (int(a), int(b)), (int(c), int(d)), (0, 255, 255), 2)
                mask = cv2.circle(mask, (int(a), int(b)), 3, (0, 255, 255), -1)
                frame = cv2.circle(frame,(int(a), int(b)), 3, (0, 255, 255), -1)
                
            result = cv2.add(frame, mask)
            

            cv2.imshow('2.2 Video tracking', result)
            if cv2.waitKey(int(1000/fps)) & 0xFF == ord('q'):
                break

            gray_1 = gray.copy()
            p0 = good_new.reshape(-1, 1, 2)

        capture.release()
        cv2.destroyAllWindows()

    def perspective_transform(self):
        logo = cv2.imread( "{}".format(self.img))
        pts_src = np.array([[0, 0], [logo.shape[1], 0], [logo.shape[1], 
                            logo.shape[0]],  [0, logo.shape[0]]], dtype=float)


        capture = cv2.VideoCapture("{}".format(self.vdieo))
        fps = capture.get(cv2.CAP_PROP_FPS)

        ret, frame = capture.read()

        while(capture.isOpened()):

            ret, frame = capture.read()
            if not ret:
                break

            dictionary = aruco.Dictionary_get(aruco.DICT_4X4_250)
            
            param = aruco.DetectorParameters_create()
            
            markerCornaers, markerIds, rejectedCandidates = aruco.detectMarkers(
                frame,
                dictionary,
                parameters = param
            )
            id1 = np.squeeze(np.where(markerIds == 1))
            id2= np.squeeze(np.where(markerIds == 2))
            id3 = np.squeeze(np.where(markerIds == 3))
            id4 = np.squeeze(np.where(markerIds == 4))

            if id1 != [] and id2 != [] and id3 != [] and id4 != []:
                pt1 = np.squeeze(markerCornaers[id1[0]])[0]
                pt2 = np.squeeze(markerCornaers[id2[0]])[1]
                pt3 = np.squeeze(markerCornaers[id3[0]])[2]
                pt4 = np.squeeze(markerCornaers[id4[0]])[3]

                pts_dst = [[pt1[0], pt1[1]]]
                pts_dst = pts_dst + [[pt2[0], pt2[1]]]
                pts_dst = pts_dst + [[pt3[0], pt3[1]]]
                pts_dst = pts_dst + [[pt4[0], pt4[1]]]
                pts_dst = np.array(pts_dst)

                im_dst = frame
                h, status = cv2.findHomography(pts_src, pts_dst)
                temp = cv2.warpPerspective(logo, h, (im_dst.shape[1], im_dst.shape[0]))
                
                cv2.fillConvexPoly(im_dst, pts_dst.astype(int), 0, 16)
                im_dst = im_dst + temp
                cv2.imshow('3.1 Perspective Transform', im_dst)
            
            if cv2.waitKey(int(1000/fps)) & 0xFF == ord('q'):
                break
        
        capture.release()
        cv2.destroyAllWindows()

    def reconstruction(self, img):
        b, g, r = cv2.split(img)

        pca = PCA(n_components=10)

        lower_dimension_b = pca.fit_transform(b)
        approximation_b = pca.inverse_transform(lower_dimension_b)

        lower_dimension_g = pca.fit_transform(g)
        approximation_g = pca.inverse_transform(lower_dimension_g)

        lower_dimension_r = pca.fit_transform(r)
        approximation_r = pca.inverse_transform(lower_dimension_r)
        
        clip_b = np.clip(approximation_b, a_min = 0, a_max = 255)
        clip_g = np.clip(approximation_g, a_min = 0, a_max = 255)
        clip_r = np.clip(approximation_r, a_min = 0, a_max = 255)
        n_img = (cv2.merge([clip_b, clip_g, clip_r])).astype(np.uint8)
        return n_img

    def image_reconstruction(self):



        fig = plt.figure(figsize=(15, 4))
        for id in range(1, 16):

            img = cv2.imread("{}/sample ({}).jpg".format(self.folder_path,id))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            n_img = self.reconstruction(img)
 
            plt.subplot(4, 15, id)
            plt.axis('off')
            plt.imshow(img)

            plt.subplot(4, 15, id + 15)
            plt.axis('off')
            plt.imshow(n_img)

        for id in range(16, 31):
            img = cv2.imread("{}/sample ({}).jpg".format(self.folder_path,id))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            n_img = self.reconstruction(img)

            plt.subplot(4, 15, id + 15)
            plt.axis('off')
            plt.imshow(img)

            plt.subplot(4, 15, id + 30)
            plt.axis('off')
            plt.imshow(n_img)

        fig.text(0, 0.9, 'Original', va='center', rotation='vertical')
        fig.text(0, 0.65, 'Reconstruction', va='center', rotation='vertical')
        fig.text(0, 0.4, 'Original', va='center', rotation='vertical')
        fig.text(0, 0.15, 'Reconstruction', va='center', rotation='vertical')
        plt.tight_layout(pad=0.5)

        plt.show()

        
    def reconstruct_err(self):
        errorList = []
        for id in range(1, 31):
            img = cv2.imread("{}/sample ({}).jpg".format(self.folder_path,id))
            n_img = self.reconstruction(img)

            img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            n_img_gray = cv2.cvtColor(n_img, cv2.COLOR_BGR2GRAY)
            n_img_gray = cv2.normalize(n_img_gray, None, 0, 255, cv2.NORM_MINMAX)
            error = np.sum(np.absolute(img_gray-n_img_gray))
            errorList.append(error)
        print(errorList)


