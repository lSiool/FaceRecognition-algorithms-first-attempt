import cv2 as cv
# Read image from your local file system
original_image = cv.imread('pexels-photo-1648387.jpeg ')
# Convert color image to grayscale for Viola-Jones
grayscale_image = cv.cvtColor(original_image, cv.COLOR_BGR2GRAY)
# Load the classifier and create a cascade object for face detection
face_cascade = cv.CascadeClassifier(r'C:/Users/Stas/faceDTest/myvenv/Lib/site-packages/cv2/data/haarcascade_frontalface_alt.xml')
# face_cascade = cv2.CascadeClassifier('C:\\opencv\\build\\etc\\haarcascades\\haarcascade_frontalface_default.xml')
# eye_cascade = cv.CascadeClassifier(r'C:/Users/Stas/faceDTest/myvenv/Lib/site-packages/cv2/data/haarcascade_eye.xml') 
detected_faces = face_cascade.detectMultiScale(grayscale_image)

print(len(grayscale_image))

for (column, row, width, height) in detected_faces:
    cv.rectangle(
        original_image,
        (column, row),
        (column + width, row + height),
        (0, 255, 0),
        2
    )
cv.imshow('Image', original_image)
cv.waitKey(0)
cv.destroyAllWindows()