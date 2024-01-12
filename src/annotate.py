"""
Author: Brian van den Berg
Description: This script uses a pre-trained Cascade model to detect faces in images or from a webcam stream, aiding in almost automatic annotation work.
"""

import argparse
import os
import platform
import sys
import numpy as np
import cv2
import copy

# Constants for Standard Input
STD_MODELS_FOLDER = "models"
STD_MODEL_FILENAMES = ['cascade.xml']
STD_OUTPUT_FOLDER = "output"
IMAGE_EXTENSIONS = ['.jpg', '.jpeg', '.png']

# Input Size for Cascade Model
CASCADE_SIZE = 720

# Constants for Webcam Stream
STREAM_CAM_ID = 0
STREAM_WIDTH = 1280
STREAM_HEIGHT = 720

# Global variables for the trained models
cascade = None

# Counter for annotations
annotation_i = 0

def init_models(models_folder: str) -> None:
    """
    Initialize face detection model.

    Parameters:
    - models_folder (str): The folder containing the pre-trained models.

    Returns:
    None
    """
    global cascade

    # Load the Cascade model
    cascade = cv2.CascadeClassifier()
    cascade.load(os.path.join(models_folder, 'cascade.xml'))

def squarify_image(image: np.ndarray) -> np.ndarray:
    """
    Adjusts the image to a square by cropping the longer axis.

    Parameters:
    - image (np.ndarray): The input image.

    Returns:
    np.ndarray: The adjusted square image.
    """
    image_width = len(image[0])
    image_height = len(image)

    # Adjust the longest axis
    if image_width > image_height:
        new_w = image_height
        difference = image_width - new_w
        new_x = round(difference / 2)
        return image[0:0 + image_height, new_x:new_x + new_w]
    else:
        new_h = image_width
        difference = image_height - new_h
        new_y = round(difference / 2)
        return image[new_y:new_y + new_h, 0:0 + image_width]

def squarify_box(box: list[int]) -> list[int]:
    """
    Turns a detected face bounding box into a square.

    Parameters:
    - box (list[int]): The list of four integer values representing the bounding box coordinates (x, y, width, height).

    Returns:
    list[int]: The adjusted bounding box coordinates to form a square or None if the adjustment is not valid.
    """
    global CASCADE_SIZE

    # Turn the detected face into a square
    new_width = box[3]
    width_difference = new_width - box[2]
    new_x = box[0] - round(width_difference / 2)
            
    # Apply the changed box if valid
    if new_x > 0 and new_x + new_width < CASCADE_SIZE:
        box[0] = new_x
        box[2] = new_width
        return box
    else:
        return None

def draw_box(image: np.ndarray, box: list[int]) -> np.ndarray:
    """
    Draws a bounding box on the input image.

    Parameters:
    - image (np.ndarray): The input image.
    - box (list[int]): The list of four integer values representing the bounding box coordinates (x, y, width, height).

    Returns:
    np.ndarray: The image with the bounding box drawn.
    """
    # Limit the box to be higher than 0
    for i, val in enumerate(box):
        if val < 0:
            box[i] = 0

    # Define bounding box rectangle
    color = (0, 255, 0)
    cv2.rectangle(image, box, color, 1)
    return image

def draw_options(image: np.ndarray) -> np.ndarray:
    """
    Draws annotation options on the input image.

    Parameters:
    - image (np.ndarray): The input image.

    Returns:
    np.ndarray: The image with annotation options drawn.
    """
    # Define the text and font parameters
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    font_thickness = 1
    font_color = (0, 0, 255)

    # Draw Option 'A'
    text = "Press 'A' to ACCEPT"
    cv2.putText(image, text, (5, 20), font, font_scale, font_color, font_thickness)

    # Draw Option 'R'
    text = "Press 'R' to REJECT"
    cv2.putText(image, text, (5, 40), font, font_scale, font_color, font_thickness)
    
    # Draw Option 'Q'
    text = "Press 'Q' to QUIT"
    cv2.putText(image, text, (5, 60), font, font_scale, font_color, font_thickness)

    # Draw a box to contain the options
    cv2.rectangle(image, [2, 2, 175, 70], font_color, 1)
    return image

def save_image_and_annotation(image: np.ndarray, path: str, annotation_file_path: str, annotation: str) -> None:
    """
    Saves an annotated image and its corresponding annotation to specified paths.

    Parameters:
    - image (np.ndarray): The image to be saved.
    - path (str): The path to save the image.
    - annotation_file_path (str): The path to the annotation file.
    - annotation (str): The annotation information to be written to the file.

    Returns:
    None
    """
    cv2.imwrite(path, image)
    with open(annotation_file_path, "a") as file:
        file.write(f'{annotation}\n')

def detect_and_handle_faces(image: np.ndarray, output_folder: str) -> int:
    """
    Detects faces using Cascade model, displays options, and handles user input.

    Parameters:
    - image (np.ndarray): The input image.
    - output_folder (str): The folder to save annotated frames.

    Returns:
    int: The number of faces detected and annotated. Returns 0 if the user chooses to discard annotations,
         -1 if the user chooses to quit the application.
    """
    global CASCADE_SIZE
    global annotation_i, cascade

    # Make the aspect ratio square
    image = squarify_image(image)
    
    # Rescale the image to be usable by the Cascade model
    scale_factor = CASCADE_SIZE / len(image)
    image = cv2.resize(image, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_AREA)

    # Start an annotation
    path = os.path.join(output_folder, f'auto_annotated_{annotation_i}.jpg')
    no_faces = 0
    boxes = []
    annotation_i += 1

    # Perform detection on the frame with Cascade model
    display_image = copy.deepcopy(image)
    faces = cascade.detectMultiScale(display_image)

    for face in faces:
        box = list(map(int, face[:4]))

        # Turn the box into a square
        box = squarify_box(box)
        if box is None:
            continue

        # Append the box to the boxes array
        boxes.append(box)

        # Draw a box around the face
        display_image = draw_box(display_image, box)
        break

    # Save the actual number of faces
    no_faces = len(boxes)

    # Draw the options on the frame
    display_image = draw_options(display_image)

    # Display the image
    cv2.imshow('Display', display_image)

    # Wait for the user to press a key
    key = cv2.waitKey(0)

    # Check if the pressed key is the left or right arrow key
    while key not in [ord('a'), ord('r'), ord('q')]:
        key = cv2.waitKey(0)

    # Handle the user's choice
    if key == ord('a'):
        print(f'Info: User ACCEPTED: "{path}"')

        # Formulate the annotation
        annotation = f'{path} {no_faces}'
        for box in boxes:
            annotation = f'{annotation} {box[0]} {box[1]} {box[2]} {box[3]}'

        # Save the annotated image
        save_image_and_annotation(image, path, os.path.join(output_folder, 'info.dat'), annotation)

        return no_faces
    elif key == ord('r'):
        print(f'Info: User REJECTED: "{path}"')
        return 0
    elif key == ord('q'):
        print(f'Info: User QUIT')
        return -1

def main_webcam(output_folder: str) -> None:
    """
    Processes frames from the webcam stream, detects and handles faces, and prints the number of annotated faces.

    Parameters:
    - output_folder (str): The folder to save the annotated frames.

    Returns:
    None
    """
    global STREAM_CAM_ID, STREAM_WIDTH, STREAM_HEIGHT

    # Check the operating system
    if platform.system() == 'Linux':
        # For Linux
        cam = cv2.VideoCapture(STREAM_CAM_ID)
    elif platform.system() == 'Windows':
        # For Windows, add the DSHOW property
        cam = cv2.VideoCapture(STREAM_CAM_ID, cv2.CAP_DSHOW)
    else:
        # Handle other operating systems if needed
        print("Error: Unsupported operating system not supported in streaming module")

    # Set the stream properties
    cam.set(cv2.CAP_PROP_FRAME_WIDTH, STREAM_WIDTH)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, STREAM_HEIGHT)
    cam.set(cv2.CAP_PROP_FPS, 60)

    # Make sure that the camera device is opened
    if not cam.isOpened():
        print(f'Error: Could not open camera with the ID {STREAM_CAM_ID}. Please try a different camera.')

    while True:
        # Fetch a frame
        ret, frame = cam.read()
        if ret:
            # Perform the annotation on the current frame
            ret = detect_and_handle_faces(frame, output_folder)
            if ret < 0:
                cv2.destroyAllWindows()
                sys.exit(0)
            
            # Print the number of annotated faces
            print(f'Info: Number of faces annotated: {ret}')
        
        # Flush the capture device
        for _ in range(3):
            cam.read()

def main_images(input_folder: str, output_folder: str) -> None:
    """
    Processes images from the input folder, detects and handles faces, and saves annotated frames to the output folder.

    Parameters:
    - input_folder (str): The folder containing input images.
    - output_folder (str): The folder to save annotated frames.

    Returns:
    None
    """
    global IMAGE_EXTENSIONS

    # List all the images in the input folder with the accepted extensions
    image_paths = [os.path.join(input_folder, f) for f in os.listdir(input_folder) if f.lower().endswith(tuple(IMAGE_EXTENSIONS))]
    for path in image_paths:
        # perform face detection
        image = cv2.imread(path)
        ret = detect_and_handle_faces(image, output_folder)
        if ret < 0:
            break

        # Print the number of annotated faces
        print(f'Info: Number of faces annotated: {ret}')
    
    # Finished running
    cv2.destroyAllWindows()
    sys.exit(0)

if __name__ == '__main__':
    # Create argument parser
    parser = argparse.ArgumentParser(description="Apply the trained models to a folder of test data.")
    
    # Add arguments
    parser.add_argument("--models", type=str, help="Path to the models folder")
    parser.add_argument("--input", type=str, help="Path to the input folder")
    parser.add_argument("--output", type=str, help="Path to the output folder")

    # Parse arguments
    args = parser.parse_args()

    # Manage models folder input
    models_folder = args.models
    if models_folder is None and os.path.exists(STD_MODELS_FOLDER):
        models_folder = STD_MODELS_FOLDER
    elif not os.path.exists(STD_MODELS_FOLDER):
        print(f'Error: Models folder not provided and {STD_MODELS_FOLDER} does not exist. Please provide a valid folder.')
        sys.exit(1)

    # Check if the expected model files exist
    missing_model_files = [filename for filename in STD_MODEL_FILENAMES if not os.path.isfile(os.path.join(models_folder, filename))]
    if missing_model_files:
        print(f'Error: The following expected model files were missing: {missing_model_files}')
        sys.exit(1)

    # Initialize the trained models
    init_models(models_folder)

    # Manage the output folder input
    output_folder = args.output
    if output_folder is None:
        output_folder = STD_OUTPUT_FOLDER
    os.makedirs(output_folder, exist_ok=True)

    # Clear the annotation file if present already
    with open(os.path.join(output_folder, 'info.dat'), "w") as file:
        file.write('')

    # Manage input folder input
    input_folder = args.input
    if input_folder is None:
        main_webcam(output_folder)
    elif os.path.exists(input_folder):
        main_images(input_folder, output_folder)
    else:
        print(f'Error: Input folder provided does not exist. Please provide a valid folder.')
        sys.exit(1)
