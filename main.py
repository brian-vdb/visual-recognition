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
    """"""

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
    
    # Manage test folder input
    test_folder = args.test
    if test_folder == None and os.path.exists(STD_TEST_FOLDER):
        test_folder = STD_TEST_FOLDER
    elif not os.path.exists(STD_TEST_FOLDER):
        print(f'Error: Test folder not provided and {STD_TEST_FOLDER} does not exist. Please provide a valid folder.')
        sys.exit(1)
    
    # Call the main
    main(models_folder, test_folder)
