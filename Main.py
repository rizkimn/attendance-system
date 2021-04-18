import cv2, os

faceCascade = cv2.CascadeClassifier('classifiers/haarcascade_frontalface_default.xml')
cap = cv2.VideoCapture(0)

name = input("Type your name : ")
faceID = input("Type your ID Number : ")

dataPath = 'dataset/' + ''.join(name.split(' ')).lower()
datasetLimit = 50

generateMode = False
generateIndex = 1
while True:
    _, frame = cap.read()
    frame = cv2.flip(frame, 1)
    grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(grayscale, 1.3, 4)
    for x,y,w,h in faces:
        frame = cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), 2)
        frame = cv2.putText(frame, name + " | " + faceID, (x, y-15), cv2.FONT_HERSHEY_DUPLEX, 0.8, (0,255,0), 1, cv2.LINE_AA)

    cv2.imshow("Testing", frame)

    if generateIndex >= datasetLimit:
        print("| Done | Dataset Generated!")
        break
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