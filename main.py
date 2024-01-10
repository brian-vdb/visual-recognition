import cv2
import os
import argparse
import sys


# Standard Constants
STD_MODELS_FOLDER = "models"
MODEL_FILENAMES = ['cascade.xml', 'mean_face.npy', 'best_eigenfaces.npy' ,'recognizer.yml']
STD_TEST_FOLDER = "test"
IMAGE_EXTENSIONS = ['.jpg', '.jpeg', '.png']

def main(models_folder: str, test_folder: str):
    # Load in the face detection haar cascade model
    cascade = cv2.CascadeClassifier()
    cascade.load(os.path.join(models_folder, 'cascade.xml'))

    # List all the images in the test folder with the accepted extensions
    image_paths = [os.path.join(test_folder, f) for f in os.listdir(test_folder) if f.lower().endswith(tuple(IMAGE_EXTENSIONS))]

    # Loop through every available image path
    for image_path in image_paths:
        image = cv2.imread(image_path)

        # Create a grayscale version of the image
        image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Applying face detection
        faces = cascade.detectMultiScale(image_gray, 1.1, 12)

        # Map the coordinates of rectangles back to the original image
        for (x, y, w, h) in faces:
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
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
