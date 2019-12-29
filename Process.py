import cv2
import numpy as np
import dlib
from matplotlib import pyplot as plt

imgsrc = cv2.imread("gallery_src/shuffle2.jpg")
img = cv2.cvtColor(imgsrc, cv2.COLOR_BGR2RGB)

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

def LiveCAP():
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = detector(grayscale)
        for face in faces:
            x1 = face.left()
            y1 = face.top()
            x2 = face.right()
            y2 = face.bottom()
            #cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

            landmarks = predictor(grayscale, face)
            for n in range(0, 68):
                x = landmarks.part(n).x
                y = landmarks.part(n).y
                cv2.circle(frame, (x, y), 6, (15, 250, 0), -1)
                #blurred = cv2.GaussianBlur(frame, (5, 5), 15)

        cv2.imshow("Live cap", frame)
        if (cv2.waitKey(1) & 0xFF) == ord('x'):
            break
    cap.release()
    cv2.destroyAllWindows()

def OnIMG():
    grayscale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = detector(grayscale)
    for face in faces:
        x1 = face.left()
        y1 = face.top()
        x2 = face.right()
        y2 = face.bottom()

        landmarks = predictor(grayscale, face)
        for n in range(0, 68):
            x = landmarks.part(n).x
            y = landmarks.part(n).y
            cv2.circle(img, (x, y), 6, (0, 82, 255), -1)
            #blurred = cv2.GaussianBlur(n, (50, 50), 30)

    plt.imshow(img)
    plt.show()

print("Nyomjon írjon 1-et, ha liveban szeretne detektalni!\nNyomjon 2-t, ha a képen!\nKerem valasszon: ")
response = input()
if response == '1':
    LiveCAP()
elif response == '2':
    OnIMG()
else:
    print("Rossz input!")