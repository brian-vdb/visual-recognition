"""
Author: Brian van den Berg
Description: Apply face recognition preprocessing to images in the input folder using annotations from 'info.dat' file.
             Save the processed faces in the output folder and generate a CSV file for face recognition labels.

Usage:
    python script_name.py --input <input_folder_path> --output <output_folder_path>

Options:
    --input: Path to the input folder containing images and 'info.dat' file. Default is 'input' folder.
    --output: Path to the output folder where processed faces and recognition CSV will be saved. Default is 'output' folder.
"""

import os
import sys
import argparse
import math
import re
import csv
import cv2

from gui.ClassAnnotationApp import run_class_annotation_app

# Standard Input Constants
STD_INPUT_FOLDER = 'input'
STD_INPUT_FILENAMES = ['info.dat']
STD_OUTPUT_FOLDER = 'output'
IMAGE_EXTENSIONS = ['.jpg', '.jpeg', '.png']
EIGENFACE_BOX_SIZE = 64

def read_boxes(annotations_path: str) -> list[dict[str, any]]:
    """
    Read face annotations from a file and return a list of dictionaries,
    where each dictionary contains information about an image and its associated face boxes.

    Parameters:
    - annotations_path (str): The path to the annotations file.

    Returns:
    - list[dict[str, any]]: A list of dictionaries, each containing:
        - 'path' (str): The path to the image.
        - 'box' (list[int]): A list representing the face box [x, y, w, h].
    """
    boxes = []
    
    # Open the annotation file
    with open(annotations_path, 'r') as file:
        for i, line in enumerate(file):
            # Split the path into values
            values = line.strip().split()

            # Check for the end of file
            if len(values) == 0:
                break

            # Get the image path and number of faces in the image
            path = values[0]
            face_i = 1

            # Error check the line to ensure it has the expected format
            try:
                num_faces = int(values[face_i])
            except ValueError:
                print(f'Line {i}: Warning: Skipping Line; Incorrect Format:')
                print(f'- Line: "{line.strip()}"')
                continue

            # Loop through every face in the picture
            for _ in range(num_faces):
                # Get the variables for face 'j' in the picture
                x = int(values[face_i + 1])
                y = int(values[face_i + 2])
                w = int(values[face_i + 3])
                h = int(values[face_i + 4])
                face_i += 4

                # Save the variables as a facebox
                boxes.append({
                    'path': path,
                    'box': [x, y, w, h]
                })

    # Return the boxes
    return boxes

def get_size_average(boxes: list[dict[str, any]]) -> tuple[int, int]:
    """
    Calculate the average size of face boxes in a list of dictionaries.

    Parameters:
    - boxes (list[dict[str, any]]): A list of dictionaries, each containing:
        - 'path' (str): The path to the image.
        - 'box' (list[int]): A list representing the face box [x, y, width, height].

    Returns:
    - tuple[int, int]: A tuple representing the average size [average_width, average_height].
    """
    total_width = 0
    total_height = 0

    # Calculate the sum of width and height for all faceboxes
    for box in boxes:
        total_width += int(box['box'][2])
        total_height += int(box['box'][3])

    # Calculate the average width and height
    average_width = round(total_width / len(boxes))
    average_height = round(total_height / len(boxes))
    print(f'Info: Average Size [h, w]: [{average_height}, {average_width}]')

    return average_width, average_height

def create_eigenfaces_csv(output_folder: str, csv_filename: str) -> None:
    """
    Create or overwrite a CSV file with header ['Filename', 'Class'] in the specified folder.

    Parameters:
    - output_folder (str): The path to the output folder.
    - csv_filename (str): The name of the CSV file.

    Returns:
    - None
    """
    # Get a list of filenames in the specified folder
    files = [f for f in os.listdir(output_folder) if f.lower().endswith(tuple(IMAGE_EXTENSIONS))]
    if len(files) == 0:
        print(f'Error: No images found in the input folder. Please provide a correct image base')
        sys.exit(1)

    # Sort files based on the numerical part of the filename
    files.sort(key=lambda x: [int(c) if c.isdigit() else c for c in re.split('(\d+)', x)])

    # Create or overwrite the CSV file
    with open(os.path.join(output_folder, csv_filename), 'w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)

        # Write header to CSV file
        csv_writer.writerow(['Filename', 'Class'])

        # Write filenames with empty second values to the CSV file
        for filename in files:
            csv_writer.writerow([os.path.join(output_folder, filename), ''])

def main_recognition(input_folder: str, output_folder: str) -> None:
    """
    Perform face recognition preprocessing on images in the input folder and save results in the output folder.

    Parameters:
    - input_folder (str): The path to the folder containing image files and the annotation file ('info.dat').
    - output_folder (str): The path to the folder where recognition results and CSV file will be saved.

    Returns:
    - None
    """
    # Read the boxes from the annotation file
    boxes = read_boxes(os.path.join(input_folder, 'info.dat'))
    w_average, h_average = get_size_average(boxes)
    
    # Define the feedback variables
    face = None
    no_faces = 0

    # Iterate through faceboxes
    for i, box in enumerate(boxes):
        # Read the image
        image = cv2.imread(box['path'])

        # Extract face box coordinates
        x, y, w, h = int(box['box'][0]), int(box['box'][1]), int(box['box'][2]), int(box['box'][3])

        # Copy the face from the image
        face = image[y:y + h, x:x + w]

        # Convert the cropped face to grayscale
        face_gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)

        # Finally scale the images to a fixed maximum window size
        scale_factor = EIGENFACE_BOX_SIZE / w if w > h else EIGENFACE_BOX_SIZE / h
        face = cv2.resize(face_gray, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_AREA)
        no_faces += 1

        # Save the grayscale face
        output_path = os.path.join(output_folder, f"recognition_input_{i}.jpg")
        cv2.imwrite(output_path, face)
    print(f'Info: Found {no_faces} faces')
    print(f'Info: Output Shape [h, w]: [{len(face)}, {len(face[0])}]')

    # Create the CSV in which the labels are stored
    csv_name = 'recognition.csv'
    create_eigenfaces_csv(output_folder, csv_name)
    run_class_annotation_app(os.path.join(output_folder, csv_name))
    

if __name__ == '__main__':
    # Create argument parser
    parser = argparse.ArgumentParser(description="Apply the trained models to a folder of test data.")
    
    # Add arguments
    parser.add_argument("--input", type=str, help="Path to the input folder")
    parser.add_argument("--output", type=str, help="Path to the output folder")

    # Parse arguments
    args = parser.parse_args()

    # Manage the input folder input
    input_folder = args.input
    if input_folder is None and os.path.exists(STD_INPUT_FOLDER):
        input_folder = STD_INPUT_FOLDER
    elif input_folder is None:
        print(f'Error: Input folder provided and {STD_INPUT_FOLDER} do not exist. Please provide a valid folder.')
        sys.exit(1)

    # Check if the expected model files exist
    missing_input_files = []
    for filename in STD_INPUT_FILENAMES:
        if not os.path.isfile(os.path.join(input_folder, filename)):
            missing_input_files.append(filename)
    if len(missing_input_files) > 0:
        print(f'Error: The following expected input files were missing: {missing_input_files}')
        sys.exit(1)

    # Manage the output folder input
    output_folder = args.output
    if output_folder is None:
        output_folder = STD_OUTPUT_FOLDER
    os.makedirs(output_folder, exist_ok=True)

    # Prepare the Face Recognition dataset
    main_recognition(input_folder, output_folder)
