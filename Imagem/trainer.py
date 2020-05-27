import cv2
import os
import numpy as np
from PIL import Image

lbph = cv2.face.LBPHFaceRecognizer_create(2, 2, 7, 7, 15)

def getImagemComId():
    path = [os.path.join('Dataset', f) for f in os.listdir('Dataset')]
    faces = []
    ids = []
    for pathImg in path:
        imagemFace = Image.open(pathImg).convert('L')
        imagemFace = imagemFace.resize((220, 220))
        imagemFace = np.array(imagemFace, 'uint8')
        id = int(os.path.split(pathImg)[1].split('_')[0].replace("G", ""))
        print(id)
        ids.append(id)
        faces.append(imagemFace)
        cv2.waitKey(10)
    return np.array(ids), faces

print("Training...")

ids, faces = getImagemComId()

lbph.train(faces, ids)
lbph.write('classifierLBPH.yml')

print("Training DONE!")
