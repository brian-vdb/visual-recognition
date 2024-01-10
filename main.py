import os
import sys
import argparse
import ctypes
import numpy as np
import cv2

# Standard Constants
STD_MODELS_FOLDER = "models"
MODEL_FILENAMES = ['cascade.xml', 'eigenface_shape.npy', 'mean_face.npy', 'best_eigenfaces.npy' ,'recognizer.yml']
STD_TEST_FOLDER = "test"
IMAGE_EXTENSIONS = ['.jpg', '.jpeg', '.png']

# Function to get the screen dimensios
def get_screen_dimensions() -> tuple[int, int]:
    user32 = ctypes.windll.user32
    screen_width, screen_height = user32.GetSystemMetrics(0), user32.GetSystemMetrics(1)
    return screen_width, screen_height

# Function to scale an image according to the screen size
def scale_image_to_screen(image: np.ndarray) -> np.ndarray:
    screen_width, screen_height = get_screen_dimensions()
    screen_width = round(screen_width * 0.5)
    screen_height = round(screen_height * 0.5)

    # Scale if the image is bigger than the screen size
    if image.shape[1] > screen_width or image.shape[0] > screen_height:
        # Calculate scaling factors for width and height
        scale_x = screen_width / image.shape[1]
        scale_y = screen_height / image.shape[0]

        # Choose the minimum scaling factor to maintain aspect ratio
        scale_factor = min(scale_x, scale_y)

        # Resize the image
        scaled_image = cv2.resize(image, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_AREA)

        return scaled_image
    else:
        return image

def main(models_folder: str, test_folder: str):
    # Load in the face detection haar cascade model
    cascade = cv2.CascadeClassifier()
    cascade.load(os.path.join(models_folder, 'cascade.xml'))

    # Load in the eigenfaces recognition model and preprocessing vectors
    recognizer = cv2.face.EigenFaceRecognizer().create()
    recognizer.read(os.path.join(models_folder, 'recognizer.yml'))
    best_eigenfaces = np.load(os.path.join(models_folder, 'best_eigenfaces.npy'))
    mean_face = np.load(os.path.join(models_folder, 'mean_face.npy'))
    eigenface_shape = tuple(np.load(os.path.join(models_folder, 'eigenface_shape.npy')))

    # List all the images in the test folder with the accepted extensions
    image_paths = [os.path.join(test_folder, f) for f in os.listdir(test_folder) if f.lower().endswith(tuple(IMAGE_EXTENSIONS))]

    # Loop through every available image path
    for image_path in image_paths:
        image = cv2.imread(image_path)
        image = scale_image_to_screen(image)

        # Create a grayscale version of the image
        image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Applying face detection
        faces = cascade.detectMultiScale(image_gray)

        # Map the coordinates of rectangles back to the original image
        for (x, y, w, h) in faces:
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 1)

            # Get the region of interest
            face = image_gray[y:y + h, x:x + w]

            # Calculate the scaling factors
            scale_x = eigenface_shape[1] / face.shape[1]
            scale_y = eigenface_shape[0] / face.shape[0]

            # Apply preprocessing to the face
            face_scaled = cv2.resize(face, None, fx=scale_x, fy=scale_y, interpolation=cv2.INTER_AREA)
            face_flattened = face_scaled.reshape(-1)
            face_normalized = face_flattened - mean_face
            face_projected = np.dot(face_normalized, best_eigenfaces.T)

            # Recognize the projected face using the trained recognizer
            label, confidence = recognizer.predict(face_projected)
            
            # Define the text and font parameters
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.5
            font_thickness = 1
            font_color = (0, 255, 0)

            # Draw the class text
            text = f'Class: {label}'
            cv2.putText(image, text, (x, y - 6), font, font_scale, font_color, font_thickness)

            # Draw the confidence text
            text = f'Confidence: {round(confidence)}'
            cv2.putText(image, text, (x, y + w + 8), font, font_scale * 0.5, font_color, font_thickness)

        # Display the result after applying the trained models
        cv2.imshow('Result', image)
        cv2.waitKey(0)

if __name__ == "__main__":
    # Create argument parser
    parser = argparse.ArgumentParser(description="Apply the trained models to a folder of test data.")
    
    # Add arguments
    parser.add_argument("--models", type=str, help="Name of the models folder")
    parser.add_argument("--test", type=str, help="Folder to get the test images from")

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
    for filename in MODEL_FILENAMES:
        if not os.path.isfile(os.path.join(models_folder, filename)):
            missing_model_files.append(filename)
    if len(missing_model_files) > 0:
        print(f'Error: The following expected model files were missing: {missing_model_files}')
        sys.exit(1)
    
    # Manage test folder input
    test_folder = args.test
    if test_folder == None and os.path.exists(STD_TEST_FOLDER):
        test_folder = STD_TEST_FOLDER
    elif not os.path.exists(STD_TEST_FOLDER):
        print(f'Error: Test folder not provided and {STD_TEST_FOLDER} does not exist. Please provide a valid folder.')
        sys.exit(1)
    
    # Call the main
    main(models_folder, test_folder)
