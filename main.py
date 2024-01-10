import cv2
import os
import argparse
import sys


# Standard Constants
STD_MODELS_FOLDER = "models"
STD_TEST_FOLDER = "test"

def main():
    """"""

if __name__ == "__main__":
    # Create argument parser
    parser = argparse.ArgumentParser(description="Apply the trained models to a folder of test data.")
    
    # Add arguments
    parser.add_argument("--models", type=str, help="Name of the models folder")
    parser.add_argument("--test", type=str, help="Name of the folder with test images")

    # Parse arguments
    args = parser.parse_args()

    # Check the models input
    models_path = args.models
    if models_path == None:
        models_path = STD_MODELS_FOLDER

    # Check the test input
    test_path = args.test
    if test_path == None:
        test_path = STD_TEST_FOLDER

    print(f'{models_path}, {test_path}')
    
    # Call the main
    main()
