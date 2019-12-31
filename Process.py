import cv2
import numpy as np
import dlib
import os
import glob
from PIL import Image
from matplotlib import pyplot as plt

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

sourcefold = "/home/gboldi19/Background_removal/gallery_src/"
destfold = "/home/gboldi19/Background_removal/gallery_dest/"

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
    os.chdir("..")
    images = []
    glob_images = glob.glob(destfold+"*.png")
    for actimg in glob_images:
        print(os.getcwd())
        '''
        file, ext = os.path.splitext(actimg)
        folder = os.path.basename(actimg)
        print(folder, file, ext)
        '''
        images.append(actimg)
    for image in images:
        imgsrc = cv2.imread(image)
        img = cv2.cvtColor(imgsrc, cv2.COLOR_BGR2RGB)
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
        plt.savefig(image)

if not os.path.exists(destfold):
    os.makedirs(destfold)

os.chdir(sourcefold)
#Bing API segítségével fetchelt kepek nagy elofordulasban .jpg kiterjesztessel rendelkeznek
for infile in glob.glob("*.jpg"):
    bonefile = os.path.basename(infile)
    file, ext = os.path.splitext(bonefile)
    print(os.getcwd(), file)
    im = Image.open(bonefile)
    rgb_im = im.convert('RGB')
    rgb_im.save(destfold + file + ".png", "PNG")
for infile in glob.glob("*.jpeg"):
    bonefile = os.path.basename(infile)
    file, ext = os.path.splitext(bonefile)
    print(os.getcwd(), file)
    im = Image.open(bonefile)
    rgb_im = im.convert('RGB')
    rgb_im.save(destfold + file + ".png", "PNG")
for infile in glob.glob("*.JPG"):
    bonefile = os.path.basename(infile)
    file, ext = os.path.splitext(bonefile)
    print(os.getcwd(), file)
    im = Image.open(bonefile)
    rgb_im = im.convert('RGB')
    rgb_im.save(destfold + file + ".png", "PNG")
for infile in glob.glob("*.JPEG"):
    bonefile = os.path.basename(infile)
    file, ext = os.path.splitext(bonefile)
    print(os.getcwd(), file)
    im = Image.open(bonefile)
    rgb_im = im.convert('RGB')
    rgb_im.save(destfold + file + ".png", "PNG")

print("Nyomjon írjon 1-et, ha liveban szeretne detektalni!\nNyomjon 2-t, ha a képen!\nKerem valasszon: ")
response = input()
if response == '1':
    LiveCAP()
elif response == '2':
    OnIMG()
else:
    print("Rossz input!")