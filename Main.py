import cv2

faceCascade = cv2.CascadeClassifier('classifiers/haarcascade_frontalface_default.xml')
cap = cv2.VideoCapture(0)

while True:
    _, frame = cap.read()
    frame = cv2.flip(frame, 1)
    grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(grayscale, 1.3, 4)
    for x,y,w,h in faces:
        cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), 2)

    cv2.imshow("Testing", frame)
    
    keyCode = cv2.waitKey(1) & 0xFF
    if keyCode == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()