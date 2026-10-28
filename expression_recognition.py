
import cv2

# Ask the user for an image path
image_path = input("Enter the image path: ")

# Load the image
image = cv2.imread(image_path)

if image is None:
    print("Could not load the image. Check the path.")
else:
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    print("Image loaded successfully!")
    print("Image dimensions:", image.shape)

    cv2.imshow("Your Image", image)
    cv2.imshow("Grayscale Image", gray)

    cv2.waitKey(0)
    cv2.destroyAllWindows()




