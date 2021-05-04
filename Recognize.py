import cv2, os

cap = cv2.VideoCapture(0)

faceCascade = cv2.CascadeClassifier("classifiers/haarcascade_frontalface_default.xml")
trainCascade = cv2.face.LBPHFaceRecognizer_create()
trainCascade.read("classifiers/trained_classifier.xml")

names = []

for user in os.listdir("dataset"):
    name = user.split("-")
    for i in range(len(name)):
        name[i] = name[i].capitalize()

    name = " ".join(name)
    names.append(name)


while True:
    _, frame = cap.read()
    frame = cv2.flip(frame, 1)
    grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(grayscale, 1.3, 4, minSize=(50,50))
    for (x,y,w,h) in faces:
        id, persent = trainCascade.predict(grayscale[y:y+h, x:x+w])

        if persent < 60:
            color = (0, 255, 0)
            text = names[len(names) - id] + " | " + str(persent) + "%"
        else:
            color = (0, 0, 255)
            text = "Unknown"

        cv2.rectangle(frame, (x,y), (x+w, y+h), color, 2)
        cv2.putText(frame, text, (x, y-15), cv2.FONT_HERSHEY_DUPLEX, 0.8, color, 1, cv2.LINE_AA)

    cv2.imshow("Face Attendance", frame)


    k = cv2.waitKey(1) & 0xFF
    if k == ord('q'): break;
cap.release()
cv2.destroyAllWindows()
