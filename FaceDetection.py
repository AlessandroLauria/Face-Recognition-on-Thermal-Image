import cv2

# Get user supplied values
image_path = "./Dataset/Alessandro (8).jpeg"
casc_path = "./haarcascade_frontalface_default.xml"

faceCascade = cv2.CascadeClassifier(casc_path)

# Read the image
image = cv2.imread(image_path)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Detect faces in the image
faces = faceCascade.detectMultiScale(
    gray,
    scaleFactor=1.1,
    minNeighbors=5,
    minSize=(30, 30)
)

print("Found {0} faces!".format(len(faces)))

# Draw a rectangle around the faces
for (x, y, w, h) in faces:
    cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)


cv2.imshow("Faces found", image)
cv2.waitKey(0)