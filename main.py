import cv2
import os
import argparse
import sys

def main():
    """"""

if __name__ == "__main__":
    # Create argument parser
    parser = argparse.ArgumentParser(description="Process faceboxes and save eigenface images.")
    
    # Add arguments
    parser.add_argument("--models", type=str, help="Name of the models folder")

    # Parse arguments
    args = parser.parse_args()

    # Check if the annotations file exists
    annotations = args.annotations
    if not os.path.isfile(annotations):
        print(f"Error: Annotations file '{annotations}' does not exist. Please provide a valid file.")
        sys.exit(1)
    
    # Call the main
    main(annotations)
