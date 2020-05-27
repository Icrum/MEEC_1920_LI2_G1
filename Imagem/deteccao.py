import cv2

#Detection method
detectorFace = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

#LBPH Features
classifier = cv2.face.LBPHFaceRecognizer_create(2,2,8,8,16)

classifier.read("classifierLBPH.yml") #classifier

font = cv2.FONT_HERSHEY_DUPLEX

largura, altura = 220, 220

camera = cv2.VideoCapture(0)

while (True):
    connected, img = camera.read()
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    facesDetected = detectorFace.detectMultiScale(imgGray, 1.3, 5)
    for (x, y, l, a) in facesDetected:
        imgFace = cv2.resize(imgGray[y:y + a, x:x + l], (largura, altura))
        cv2.rectangle(img, (x, y), (x + l, y + a), (0,0,255), 2)
        id, confidence = classifier.predict(imgFace) #cropped image
        nome = ""
        if id == 1:
            nome = 'Claudia'
        else:
            nome = 'Not Claudia'

        cv2.putText(img, nome, (x,y +(a+30)), font, 1 , (255, 0, 133))
        cv2.putText(img, str(round(confidence,2)), (x,y + (a+70)), font, 1 , (255 ,0 ,0))

    cv2.imshow("Recognition LBPH", img)
    if cv2.waitKey(1) == ord('q'):
        break

camera.release()
cv2.destroyAllWindows()