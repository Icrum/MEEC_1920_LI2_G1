import sys
from PyQt5 import QtWidgets, uic, QtGui
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import QTimer, qVersion

import numpy as nd

import cv2
import os
from PIL import Image
import numpy as np
from PyQt5.QtWidgets import QFileDialog
from sklearn.model_selection import train_test_split
from sklearn import metrics
import pandas as pd
import matplotlib.pyplot as plt

classifier = cv2.face.LBPHFaceRecognizer_create()

classifier.read("classifierLBPH.yml") #classifier

"""
def img2pixmap(image):
    height, width, channel = image.shape
    bytesPerLine = channel * width
    qimage = QImage(image.data, width, height, bytesPerLine, QImage.Format_BGR888)
    pixmap = QPixmap.fromImage(qimage)
    return pixmap
"""

# Reconhecimento pela Webcam
def rec_lbph_window():

    if not camera.isOpened():
        camera.open(0)
        window.video.setText("Turning Camera ON")

    # Detection method
    detectorFace = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

    font = cv2.FONT_HERSHEY_DUPLEX

    largura, altura = 220, 220

    #camera = cv2.VideoCapture(0)

    while (True):
        connected, img = camera.read()
        imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        facesDetected = detectorFace.detectMultiScale(imgGray, 1.3, 5)
        for (x, y, l, a) in facesDetected:
            imgFace = cv2.resize(imgGray[y:y + a, x:x + l], (largura, altura))
            cv2.rectangle(img, (x, y), (x + l, y + a), (0, 0, 255), 2)
            id, confidence = classifier.predict(imgFace)  # cropped image
            nome = ""
            if id == 1:
                nome = 'Claudia'
            else:
                nome = 'Not Claudia'

            cv2.putText(img, nome, (x, y + (a + 30)), font, 1, (255, 0, 133))
            cv2.putText(img, str(round(confidence, 2)), (x, y + (a + 70)), font, 1, (255, 0, 0))

        cv2.imshow("Recognition LBPH", img)
        if cv2.waitKey(1) == ord('q'):
            break


    camera.release()
    cv2.destroyAllWindows()
    return img
    #self.window.labelFrameOutput.setPixmap(img2pixmap(image))

def start_clicked():
    window.video.setText("Reconhecimento em andamento")
    #qtimerFrame.start(50)
    rec_lbph_window()


"""
def on_cameraOFF_clicked():
    qtimerFrame.stop()
    if camera.isOpened():
        camera.release()
    window.labelText.setText("Turning Camera OFF")
"""

def stop_clicked():
    window.close()


camera = cv2.VideoCapture(0)
app = QtWidgets.QApplication(sys.argv)
window = uic.loadUi("mainwindow.ui")
window.ButtonStart.clicked.connect(start_clicked)
window.ButtonStop.clicked.connect(stop_clicked)


window.video.setScaledContents(True)

window.show()
app.exec()
