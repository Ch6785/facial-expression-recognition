
import cv2
import joblib
import os

# Load our trained model
model = joblib.load("emotion_model.pkl")

# Load OpenCV's face detector
cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
face_detector = cv2.CascadeClassifier(cascade_path)

# Ask for an image
image_path = input("Enter the image path: ").strip().strip('"')

image = cv2.imread(image_path)

if image is None:
    print("Error: Could not open the image.")
    raise SystemExit

# Detect faces
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
faces = face_detector.detectMultiScale(
    gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
)

if len(faces) == 0:
    print("No face detected. Try a clear, front-facing photo.")
else:
    for (x, y, w, h) in faces:
        face = gray[y:y+h, x:x+w]
        face = cv2.resize(face, (48, 48))
        features = face.flatten().reshape(1, -1) / 255.0

        emotion = model.predict(features)[0]

        cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(
            image, emotion.upper(), (x, max(y-10, 20)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2
        )

        print("Predicted expression:", emotion.upper())

# Show and save the result
cv2.imshow("Expression Prediction", image)
cv2.imwrite("prediction_result.jpg", image)

print("Result saved as prediction_result.jpg")
cv2.waitKey(0)
cv2.destroyAllWindows()
