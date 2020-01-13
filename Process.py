import cv2
import numpy as np
import dlib
import os
import glob
from PIL import Image
from matplotlib import pyplot as plt

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

sourcefold = "/home/gboldi19/Background_removal/gallery_src/"
destfold = "/home/gboldi19/Background_removal/gallery_dest/"

def face_blur(bw, color):
    faces = cascade.detectMultiScale(bw, 1.1, 4)
    for (x, y, w, h) in faces:
        roi_color = color[y:y + h, x:x + w]
        blur = cv2.GaussianBlur(roi_color, (101, 101), 0)
        color[y:y + h, x:x + w] = blur
    return color

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
        #print(os.getcwd())
        #print(len(actimg))
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
        plt.axis("off")
        plt.savefig(image)
    print("A talált arcok: ", len(faces), "/", len(images), "képen.")
    cv2.destroyAllWindows()

def conversion():
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


if not os.path.exists(destfold):
    os.makedirs(destfold)

os.chdir(sourcefold)
print("-----Arc detektalo es homalyosito program-----")
print("--------------------Menu:---------------------")
print("1-es gomb: -> Arcelek detektalasa\n2-es gomb: -> Kepen valo bejaras\n3-as gomb: -> Arc torzitasa\n4-es gomb: -> Kepeken talalhato arcok torzitasa")
#elso futtataskor ajanlott hasznalni
#conversion()
response = input()
if response == '1':

    LiveCAP()

elif response == '2':

    OnIMG()

elif response == '3':

    video_cap = cv2.VideoCapture(0)
    while True:
        _, color = video_cap.read()
        bw = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)
        blur = face_blur(bw, color)
        cv2.imshow('Video', blur)
        if cv2.waitKey(1) & 0xFF == ord('x'):
            break

    video_cap.release()
    cv2.destroyAllWindows()

elif response == '4':

    os.chdir("..")
    images = []
    glob_images = glob.glob(destfold+"*.png")
    for actimg in glob_images:
        images.append(actimg)
    for image in images:
        imgsrc = cv2.imread(image)
        bw = cv2.cvtColor(imgsrc, cv2.COLOR_BGR2GRAY)
        blur = face_blur(bw, imgsrc)
        plt.imshow(blur)
        plt.axis("off")
        plt.savefig(image)
else:
    print("Rossz input!")