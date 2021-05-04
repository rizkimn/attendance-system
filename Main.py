from os.path import isdir
from PIL import Image
import cv2, os, numpy as np

faceCascade = cv2.CascadeClassifier('classifiers/haarcascade_frontalface_default.xml')
recognizer = cv2.face.LBPHFaceRecognizer_create()

cap = cv2.VideoCapture(0)

name = input("Type your name : ")
faceID = input("Type your ID Number : ")

dataPath = 'dataset/' + '-'.join(name.split(' ')).lower()
datasetLimit = 50

generateMode = False
generateIndex = 1

trainMode = False

while True:
    _, frame = cap.read()
    frame = cv2.flip(frame, 1)
    grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(grayscale, 1.3, 4)
    for x,y,w,h in faces:
        frame = cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), 2)
        frame = cv2.putText(frame, name + " | " + faceID, (x, y-15), cv2.FONT_HERSHEY_DUPLEX, 0.8, (0,255,0), 1, cv2.LINE_AA)

    cv2.imshow("Testing", frame)

    # Limit Dataset
    if generateIndex > datasetLimit:
        trainMode = True

        print("==========================================")
        print("| Done | Dataset Generated!")
        print("==========================================")

        break

    # Generate Dataset
    if generateMode:
        if not isdir(dataPath):
            os.mkdir(dataPath)

        cv2.imwrite(dataPath + "/dataset-" + faceID + "-" + str(generateIndex) + ".jpg", frame)
        print("| Wait | Dataset Generating ...")
        generateIndex += 1
    
    keyCode = cv2.waitKey(1) & 0xFF

    if keyCode == ord('g') : generateMode = True;
    if keyCode == ord('q') : break;


cap.release()
cv2.destroyAllWindows()

print('\n')


# Train Dataset
if trainMode:
    def getSample():
        print("| WAIT | Collecting Dataset ...")
        users = []
        images = []

        for name in os.listdir('dataset'):
            users.append(name)

        for user in users:
            for img in os.listdir('dataset/{}'.format(user)):
                imgPath = os.path.join('dataset/{}'.format(user), img)
                images.append(imgPath)

        faceSamples = []
        faceIDs = [] 

        for imgPath in images:
            imgPIL = Image.open(imgPath).convert('L')
            sampleImg = np.array(imgPIL, 'uint8')
            faceID = int(os.path.split(imgPath)[-1].split('-')[1])

            faces = faceCascade.detectMultiScale(sampleImg)
            for (x,y,w,h) in faces:
                faceSamples.append(sampleImg[y:y+h, x:x+w])
                faceIDs.append(faceID)

        faceIDs = np.array(faceIDs)
        print("| DONE | Dataset Collected !")

        return faceSamples, faceIDs

    faceSample, faceId = getSample()

    print("| WAIT | Dataset Training ...")
    recognizer.train(faceSample, faceId)
    recognizer.save('classifiers/trained_classifier.xml')

    print("==========================================")
    print("| DONE | Dataset Trained !")
    print("==========================================")
