import cv2
import numpy as np
import dlib
import face_recognition as fr
import os
import face_recognition
import time
from time import sleep

def get_encoded_faces():
    """
    looks through the faces folder and encodes all
    the faces

    :return: dict of (name, image encoded)
    """
    encoded = {}

    for dirpath, dnames, fnames in os.walk("./faces"):
        for f in fnames:
            if f.endswith(".jpg") or f.endswith(".png"):
                face = fr.load_image_file("faces/" + f)
                encoding = fr.face_encodings(face)[0]
                encoded[f.split(".")[0]] = encoding

    return encoded

def unknown_image_encoded(img):
    """
    encode a face given the file name
    """
    face = fr.load_image_file("faces/" + img)
    encoding = fr.face_encodings(face)[0]

    return encoding

def classify_face(): #im
    """
    will find all of the faces in a given image and label
    them if it knows what they are

    :param im: str of file path
    :return: list of face names
    """
    # faces = get_encoded_faces()
    # faces_encoded = list(faces.values())
    # known_face_names = list(faces.keys())

    # img = cv2.imread(im, 1)
    # #img = cv2.resize(img, (0, 0), fx=0.5, fy=0.5)
    # #img = img[:,:,::-1]

    # face_locations = face_recognition.face_locations(img)

    # unknown_face_encodings = face_recognition.face_encodings(img, face_locations)

    # face_names = []
    # for face_encoding in unknown_face_encodings:
    #     # See if the face is a match for the known face(s)
    #     matches = face_recognition.compare_faces(faces_encoded, face_encoding)
    #     name = "Unknown"

    #     # use the known face with the smallest distance to the new face
    #     face_distances = face_recognition.face_distance(faces_encoded, face_encoding)
    #     best_match_index = np.argmin(face_distances)
    #     if matches[best_match_index]:
    #         name = known_face_names[best_match_index]

    #     face_names.append(name)    
        
    #     for (top, right, bottom, left), name in zip(face_locations, face_names):
    #         # Draw a box around the face
    #         cv2.rectangle(img, (left-20, top-20), (right+20, bottom+20), (255, 0, 0), 2)

    #         # Draw a label with a name below the face
    #         cv2.rectangle(img, (left-20, bottom -15), (right+20, bottom+20), (255, 0, 0), cv2.FILLED)
    #         font = cv2.FONT_HERSHEY_DUPLEX
    #         cv2.putText(img, name, (left -20, bottom + 15), font, 1.0, (255, 255, 255), 2)

    # Display the resulting image
    cap = cv2.VideoCapture(0)
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
    
    faces = get_encoded_faces()
    faces_encoded = list(faces.values())
    known_face_names = list(faces.keys())

    while True:
        # cv2.imshow('Video', img)
        _, frame = cap.read()
        # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
        rgb_frame = frame[:, :, ::-1]
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        face_locations = detector(gray)
        # print(face_locations)

        if len(face_locations)>=1:
            for face in face_locations:

        #         # img = cv2.imread(im, 1)
        #         #img = cv2.resize(img, (0, 0), fx=0.5, fy=0.5)
        #         #img = img[:,:,::-1]

                # face_locations = face_recognition.face_locations(img)
                # print(face)
                a = []
                a.append((face.top(),face.right(),face.bottom(),face.left()))
                print(a)
                
                unknown_face_encodings = face_recognition.face_encodings(rgb_frame, a)
                # print(unknown_face_encodings)
                
                face_names = []
                for face_encoding in unknown_face_encodings:
                    # See if the face is a match for the known face(s)
                    matches = face_recognition.compare_faces(faces_encoded, face_encoding)
                    name = "Unknown"

                    # use the known face with the smallest distance to the new face
                    face_distances = face_recognition.face_distance(faces_encoded, face_encoding)
                    best_match_index = np.argmin(face_distances)
                    if matches[best_match_index]:
                        name = known_face_names[best_match_index]

                    face_names.append(name)    
                     
                    for (top, right, bottom, left), name in zip(a , face_names): #face_locations
                        # Draw a box around the face
                        cv2.rectangle(frame, (left, top), (right, bottom), (255, 0, 0), 2)

                        # Draw a label with a name below the face
                        cv2.rectangle(frame, (left, bottom), (right, bottom), (255, 0, 0), cv2.FILLED)
                        font = cv2.FONT_HERSHEY_DUPLEX
                        cv2.putText(frame, name, (left, bottom), font, 1.0, (255, 255, 255), 2)

        cv2.imshow("Frame", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            return face_names

print(classify_face())


# cap = cv2.VideoCapture(0)

# detector = dlib.get_frontal_face_detector()
# predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# while True:
#   _, frame = cap.read()
#   gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

#   faces = detector(gray)
#   # if faces ==rectangles[]
#   if len(faces)>=1:
#       # print(int(time.time()))
#       for face in faces:
#           x1 = face.left()
#           y1 = face.top()
#           x2 = face.right()
#           y2 = face.bottom()

#           cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 3) #(what , (cord1,cord2), (cord3,cord4),(color),thickness)
#           # print(face)
#           landmarks = predictor(gray, face)
                
#           for n in range(0,68):
#               x = landmarks.part(n).x
#               y = landmarks.part(n).y
#               cv2.circle(frame, (x,y), 4, (255,0,0), -1)

#       # cv2.imshow("Frame", frame)
#   else:
#       print("No faces")
#       x1 = face.left()
#       y1 = face.top()
#       x2 = face.right()
#       y2 = face.bottom()

#       cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 3)
        
#   cv2.imshow("Frame", frame)


#   key = cv2.waitKey(1)
#   if cv2.waitKey(1) & 0xFF == ord('q'):
#       break
