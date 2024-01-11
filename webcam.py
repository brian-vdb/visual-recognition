import os
import sys
import time
import argparse
import cv2
import numpy as np

# Standard Input Constants
STD_MODELS_FOLDER = "models"
STD_MODEL_FILENAMES = ['cascade.xml', 'eigenface_shape.npy', 'mean_face.npy', 'best_eigenfaces.npy' ,'recognizer.yml']

# Define the width and height
STREAM_WIDTH = 640
STREAM_HEIGHT = 480

def prepare_face(face: np.ndarray, target_shape: tuple[int, int]) -> np.ndarray:
    # Convert the face to grayscale
    face_gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)

    # Calculate the scaling factors
    scale_x = target_shape[1] / face_gray.shape[1]
    scale_y = target_shape[0] / face_gray.shape[0]

    # Prepare the image for preprocessing
    face_scaled = cv2.resize(face_gray, None, fx=scale_x, fy=scale_y, interpolation=cv2.INTER_AREA)
    face_normalized = ((face_scaled - face_scaled.min()) / (face_scaled.max() - face_scaled.min())) * 255
    image_blurred = cv2.GaussianBlur(face_normalized, (3, 3), 0)

    # Return the image as an ndarray
    return image_blurred.astype(np.uint8)

def apply_eigenfaces(face: np.ndarray, mean: np.ndarray, best_eigenfaces: np.ndarray) -> np.ndarray:
    face_flattened = face.reshape(-1)
    face_normalized = face_flattened - mean
    return np.dot(face_normalized, best_eigenfaces.T)

def main(models_folder: str):
    # Load in the face detection haar cascade model
    cascade = cv2.CascadeClassifier()
    cascade.load(os.path.join(models_folder, 'cascade.xml'))

    # Load in the eigenfaces recognizer
    recognizer = cv2.face.EigenFaceRecognizer().create()
    recognizer.read(os.path.join(models_folder, 'recognizer.yml'))

    # Load in the preprocessing information
    best_eigenfaces = np.array(np.load(os.path.join(models_folder, 'best_eigenfaces.npy')))
    face_mean = np.array(np.load(os.path.join(models_folder, 'mean_face.npy')))
    target_shape = tuple(np.load(os.path.join(models_folder, 'eigenface_shape.npy')))

    # Initialize the webcam stream
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    # Set the stream properties
    codec = 0x47504A4D  # MJPG
    cap.set(cv2.CAP_PROP_FOURCC, codec)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, STREAM_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, STREAM_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, 60.0)

    while True:
        # Get a frame
        ret, frame = cap.read()
        if ret:
            faces = cascade.detectMultiScale(frame)
            for (x, y, w, h) in faces:
                eigenfaces_time_start = time.time()

                # Get the cropped out face
                face = np.array(frame[y:y + h, x:x + w], dtype=np.uint8)

                # Prepare the face for eigenface preprocessing
                face_prepared = prepare_face(face, target_shape)

                # Project the face into the eigenvector space
                face_projected = apply_eigenfaces(face_prepared, face_mean, best_eigenfaces)

                # Recognize the projected face using the recognizer
                label, confidence = recognizer.predict(face_projected)
                eigenfaces_time_stop = time.time()

                if confidence < 3000:
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 1)

                    # Define the text and font parameters
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    font_scale = 0.5
                    font_thickness = 1
                    font_color = (0, 255, 0)

                    # Draw the class text
                    text = f'Class: {label}'
                    cv2.putText(frame, text, (x, y - 6), font, font_scale, font_color, font_thickness)

                    # Draw the confidence text
                    text = f'Confidence: {round(confidence)}, RTF: {(eigenfaces_time_stop - eigenfaces_time_start) * 1000:.2f}ms'
                    cv2.putText(frame, text, (x, y + w + 8), font, font_scale * 0.5, font_color, font_thickness)
                else:
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 1)

            # Display the Result
            cv2.imshow("Display", frame)

        # Check for exit key 'q'
        key = cv2.waitKey(1)
        if key == ord("q"):
            break

    # Clean the application
    cv2.destroyAllWindows()

if __name__ == '__main__':
    # Create argument parser
    parser = argparse.ArgumentParser(description="Apply the trained models to a folder of test data.")
    
    # Add arguments
    parser.add_argument("--models", type=str, help="Name of the models folder")

    # Parse arguments
    args = parser.parse_args()

    # Manage models folder input
    models_folder = args.models
    if models_folder == None and os.path.exists(STD_MODELS_FOLDER):
        models_folder = STD_MODELS_FOLDER
    elif not os.path.exists(STD_MODELS_FOLDER):
        print(f'Error: Models folder not provided and {STD_MODELS_FOLDER} does not exist. Please provide a valid folder.')
        sys.exit(1)

    # Check if the expected model files exist
    missing_model_files = []
    for filename in STD_MODEL_FILENAMES:
        if not os.path.isfile(os.path.join(models_folder, filename)):
            missing_model_files.append(filename)
    if len(missing_model_files) > 0:
        print(f'Error: The following expected model files were missing: {missing_model_files}')
        sys.exit(1)

    main(models_folder)
