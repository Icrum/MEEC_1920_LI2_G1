import sys
import soundfile as sf
from PyQt5 import QtWidgets, uic, QtGui
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import QTimer, qVersion
from Previsao.previsao_som import AudioHandler
import numpy as nd

import cv2
import time
import os
from PIL import Image
import numpy as np
from PyQt5.QtWidgets import QFileDialog
from sklearn.model_selection import train_test_split
from sklearn import metrics
# import pandas as pd
import matplotlib.pyplot as plt

classifier = cv2.face.LBPHFaceRecognizer_create()

classifier.read("E:\GoogleDrive\MestradoEEC\LabInt_2\MEEC_1920_LI2_G1\Imagem\classifierLBPH.yml") #classifier
ah = AudioHandler()

def img2pixmap(image):
    height, width, channel = image.shape
    bytesPerLine = channel * width
    qimage = QImage(image.data, width, height, bytesPerLine, QImage.Format_BGR888)
    pixmap = QPixmap.fromImage(qimage)
    return pixmap

def img3pixmap(image):
    # height, width, channel = image.shape
    # bytesPerLine = channel * width
    # qimage = QImage(image, 400, 100, QImage.Format_BGR888)
    pixmap = QPixmap.fromImage(image)
    return pixmap


# Reconhecimento pela Webcam
def rec_lbph_window():

    if not camera.isOpened():
        camera.open(0)
        window.video.setText("Turning Camera ON")

    # Detection method
    detectorFace = cv2.CascadeClassifier("Imagem/haarcascade_frontalface_default.xml")

    font = cv2.FONT_HERSHEY_DUPLEX

    largura, altura = 220, 220

    # Inicia o stream de audio
    ah.start()

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

        window.Video.setPixmap(img2pixmap(img))
        # data, samplerate = sf.read("Previsao/temp.wav")
        # # Pxx, freqs, bins, im = plt.specgram(data, NFFT=2048, Fs=samplerate, noverlap=900)
        # graph = plt.plot(data)
        # window.graph1.setPixmap(graph)
        # # cv2.imshow("Recognition LBPH", img)

        # Previsão Som
        grupo, comando = ah.mainloop()
        if (type(grupo) == str) and grupo != " ":
            window.Pessoa.setText(grupo)
        if (type(comando) == str) and comando != " ":
            window.Comando.setText(comando)

        time.sleep(0.1)



        if cv2.waitKey(1) == ord('q'):
            break


    camera.release()
    cv2.destroyAllWindows()
    window.Video.setPixmap(img2pixmap(img))

    # return img

def start_clicked():
    window.Video.setText("Reconhecimento em andamento")
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


window.Video.setScaledContents(True)

window.show()
app.exec()
