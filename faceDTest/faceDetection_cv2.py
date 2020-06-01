# import libraries
import cv2
import matplotlib.pyplot as plt
import cvlib as cv

image_path = 'class.png'
im = cv2.imread(image_path)
plt.imshow(im)
plt.show()

faces, confidences = cv.detect_face(im)
# loop through detected faces and add bounding box

for face in faces:
    (startX,startY) = face[0],face[1]
    (endX,endY) = face[2],face[3]
    # draw rectangle over face
    cv2.rectangle(im, (startX,startY), (endX,endY), (0,255,0), 2)
# display output        
plt.imshow(im)
plt.show()