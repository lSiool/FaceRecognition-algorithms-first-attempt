import cv2
import numpy as np
import dlib
import os
import time


cap = cv2.VideoCapture("small.mp4")

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

def rescale_frame(frame, percent=75):
    width = int(frame.shape[1] * percent/ 100)
    height = int(frame.shape[0] * percent/ 100)
    dim = (width, height)
    return cv2.resize(frame, dim, interpolation =cv2.INTER_AREA)

while True:
    _, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame75 = rescale_frame(frame, percent=50)
    faces = detector(gray)
    # if faces ==rectangles[]
    if len(faces)>=1:
        # print(int(time.time()))
        for face in faces:
            x1 = face.left()
            y1 = face.top()
            x2 = face.right()
            y2 = face.bottom()
            cv2.rectangle(frame75, (x1,y1), (x2,y2), (0,255,0), 3) #(what , (cord1,cord2), (cord3,cord4),(color),thickness)
            # print(type(face))
            landmarks = predictor(gray, face)
            print(landmarks.part(1))

            for n in range(0,68):
                x = landmarks.part(n).x
                y = landmarks.part(n).y
                cv2.circle(frame75, (x,y), 1, (255,0,0), -1)
    else:
        try:
            print("No faces")
            x1 = face.left()
            y1 = face.top()
            x2 = face.right()
            y2 = face.bottom()

            cv2.rectangle(frame75, (x1,y1), (x2,y2), (0,255,0), 3)
        except:
            pass
    cv2.imshow("Frame", frame75)

    key = cv2.waitKey(1)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break



