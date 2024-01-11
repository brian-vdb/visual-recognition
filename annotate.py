import argparse
import os
import sys
import numpy as np
import cv2
import copy

# Standard Input Constants
STD_MODELS_FOLDER = "models"
STD_MODEL_FILENAMES = ['yunet.onnx']
STD_OUTPUT_FOLDER = "output"

# Yunet Input Size
YUNET_WIDTH = 500
YUNET_HEIGHT = 500

# Stream Size
STREAM_WIDTH = 1280
STREAM_HEIGHT = 720

# Counter for annotations
annotation_i = 0

def squarify_image(image: np.ndarray) -> np.ndarray:
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
    # Turn the detected face into a square
    new_width = box[3]
    width_difference = new_width - box[2]
    new_x = box[0] - round(width_difference / 2)
            
    # Apply the changed box
    if new_x > 0 and new_x + new_width < YUNET_WIDTH:
        box[0] = new_x
        box[2] = new_width
        return box
    else:
        return None

def draw_box(image: np.ndarray, box: list[int]) -> np.ndarray:
    # Limit the box to be higher than 0
    for i, val in enumerate(box):
        if val < 0:
            box[i] = 0

    # Define bounding box rectangle
    color = (0, 255, 0)
    cv2.rectangle(image, box, color, 1)
    return image

def draw_options(image: np.ndarray) -> np.ndarray:
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

def save_image_and_annotation(image: np.ndarray, annotation: str):
    """"""

def detect_and_handle_faces(yunet: cv2.FaceDetectorYN, image: np.ndarray, output_folder: str) -> int:
    # Make the aspect ratio square
    image = squarify_image(image)
    
    # Rescale the image to be usable by the YuNet model
    scale_factor = YUNET_HEIGHT / len(image)
    image = cv2.resize(image, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_AREA)

    # Start an annotation
    global annotation_i
    path = os.path.join(output_folder, f'auto_annotated_{annotation_i}.jpg')
    no_faces = 0
    boxes = []
    annotation_i += 1

    # Perform detection on the frame with YuNet
    display_image = copy.deepcopy(image)
    _, faces = yunet.detect(display_image)
    # Draw rectangles
    if faces is not None:
        # Save the actual number of faces
        no_faces = len(faces)

        # Handle every face
        for face in faces:
            box = list(map(int, face[:4]))

            # Turn the box into a square
            box = squarify_box(box)
            if box == None:
                continue

            # Append the box to the boxes array
            boxes.append(box)

            # Draw a box around the face
            display_image = draw_box(display_image, box)

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
        print("User pressed 'a' key.")

        # Formulate the annotation
        annotation = f'{path} {no_faces}'
        for box in boxes:
            annotation = f'{annotation} {box[0]} {box[1]} {box[2]} {box[3]}'

        # Save the annotated image
        save_image_and_annotation(image, annotation)

        return no_faces
    elif key == ord('r'):
        print("User pressed 'r' key.")
        return 0
    elif key == ord('q'):
        # Clean the application
        cv2.destroyAllWindows()
        return -1

def process_webcam(yunet: cv2.FaceDetectorYN, output_folder: str):
    # Initialize the webcam stream
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    # Set the stream properties
    codec = 0x47504A4D  # MJPG
    cap.set(cv2.CAP_PROP_FOURCC, codec)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, STREAM_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, STREAM_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, 60)

    while True:
        # Fetch a frame
        ret, frame = cap.read()
        if ret:
            # Perform the annotation on the current frame
            ret = detect_and_handle_faces(yunet, frame, output_folder)
            if ret < 0:
                sys.exit(1)
            
            # Print the number of annotated faces
            print(f'Number of faces annotated: {ret}')
        
        # Flush the capture device
        for _ in range(3):
            cap.read()

def process_images(yunet: cv2.FaceDetectorYN, input_folder: str, output_folder: str):
    """"""

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

    # Initialize the YuNet model
    yunet_model_path = os.path.join(models_folder, 'yunet.onnx')
    yunet = cv2.FaceDetectorYN.create(
        model=yunet_model_path,
        config="",
        input_size=(YUNET_WIDTH, YUNET_HEIGHT)
    )

    # Manage the output folder input
    output_folder = args.output
    if output_folder == None:
        output_folder = STD_OUTPUT_FOLDER
    os.makedirs(output_folder, exist_ok=True)

    # Manage input folder input
    input_folder = args.input
    if input_folder == None:
        process_webcam(yunet, output_folder)
    elif os.path.exists(input_folder):
        process_images(yunet, input_folder, output_folder)
    else:
        print(f'Error: Input folder provided does not exist. Please provide a valid folder.')
        sys.exit(1)
