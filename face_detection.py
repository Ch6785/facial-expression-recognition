
import cv2

# Load the face detector included with OpenCV
cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
face_detector = cv2.CascadeClassifier(cascade_path)

# Ask for an image
image_path = input("Enter the full path of your image: ")
image = cv2.imread(image_path)

if image is None:
    print("Image not found. Check the path.")
else:
    # Face detection works on grayscale images
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Find faces
    faces = face_detector.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
    )

    print("Number of faces detected:", len(faces))

    # Draw a rectangle around each detected face
    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

    cv2.imshow("Face Detection", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
