import cv2
import os
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn import metrics

DATASET_PATH = r"C:\Users\perei\PycharmProjects\MEEC_1920_LI2_G1\Imagem\Dataset"
lbph = cv2.face.LBPHFaceRecognizer_create()

def getImageWithId():
    path = [os.path.join('Dataset', f) for f in os.listdir('Dataset')]
    faces = []
    ids = []
    for pathImg in path:
        imgFace = Image.open(pathImg).convert('L')  # gray
        #imgFace = imgFace.resize((220, 220))
        imgNP = np.array(imgFace, 'uint8')
        id = int(os.path.split(pathImg)[1].split('_')[0].replace("G", ""))
        #print(id)
        faces.append(imgNP)
        ids.append(id)
    return np.array(ids), faces


ids, faces = getImageWithId()

print("Training...")

Y = ids
X = np.array(faces)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=1)

lbph.train(X_train, Y_train)
lbph.write('classifierLBPH.yml')
print("Training DONE!")

Y_predict = np.zeros((len(Y_test),), dtype=int)
for i in range(len(X_test)):
    a = lbph.predict(X_test[i, :, :])
    Y_predict[i] = a[0]

cm = metrics.confusion_matrix(Y_test, Y_predict, labels=[1, 2, 3, 4, 5, 6, 7, 8])
print("\nConfusion Matrix:")
print(cm)

cr = metrics.classification_report(Y_test, Y_predict)
print("\nClassification Report:")
print(cr)
