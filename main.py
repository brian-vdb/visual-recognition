import cv2
import numpy as np

# Initialize the Camera
camera = cv2.VideoCapture(0)

codec = 0x47504A4D  # MJPG
camera.set(cv2.CAP_PROP_FPS, 60.0)
camera.set(cv2.CAP_PROP_FOURCC, codec)
camera.set(cv2.CAP_PROP_FRAME_WIDTH, 360)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 640)

# Initialize the classifier
cascade = cv2.CascadeClassifier()
cascade.load('cascade.xml')

# Check if the webcam is opened correctly
if not camera.isOpened():
    raise IOError("Cannot open webcam")

while True:
    ret, frame = camera.read()

    # Scale the image to a width of 160 pixels while maintaining the aspect ratio
    scale_factor = 160.0 / frame.shape[1]
    new_height = int(frame.shape[0] * scale_factor)
    small_frame = cv2.resize(frame, (160, new_height))

    # Convert the scaled frame to grayscale
    gray = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)

    # Applying the face detection method on the grayscale image
    faces = cascade.detectMultiScale(gray, 1.1, 9)

    # Map the coordinates of rectangles back to the original image
    for (x, y, w, h) in faces:
        x_orig = int(x / scale_factor)
        y_orig = int(y / scale_factor)
        w_orig = int(w / scale_factor)
        h_orig = int(h / scale_factor)
        cv2.rectangle(frame, (x_orig, y_orig), (x_orig+w_orig, y_orig+h_orig), (0, 255, 0), 2)

    # Display the original image with rectangles drawn
    cv2.imshow('Input', frame)

    c = cv2.waitKey(1)
    if c == 27:
        break

camera.release()
cv2.destroyAllWindows()
