import os
import argparse
import sys
import cv2

# Output Variables
STD_OUTPUT_FOLDER = "output"

def read_faceboxes(annotations_path: str) -> list[dict[str, any]]:
    """
    Read faceboxes information from an annotations file.

    Parameters:
    - annotations_path (str): Path to the annotations file.

    Returns:
    - List[Dict[str, any]]: List of dictionaries representing faceboxes.
    """
    faceboxes = []

    with open(annotations_path, 'r') as file:
        # Loop through the lines in the annotation file
        for i, line in enumerate(file):
            # Split the path into values
            values = line.strip().split()

            # Check for the end of file
            if len(values) == 0:
                print(f'Line {i}: Reached End Of File, Saved {len(faceboxes)} faces')
                break

            # Get the image path and number of faces in the image
            img_path = values[0]
            face_index = 1

            # Error check the line to ensure it has the expected format
            try:
                num_faces = int(values[face_index])
            except ValueError:
                print(f'Line {i}: Warning: Skipping Line; Incorrect Format:')
                print(f'- Line: "{line.strip()}"')
                continue

            # Loop through every face in the picture
            for _ in range(num_faces):
                # Get the variables for face 'j' in the picture
                face_x = int(values[face_index + 1])
                face_y = int(values[face_index + 2])
                face_w = int(values[face_index + 3])
                face_h = int(values[face_index + 4])
                face_index += 4

                # Save the variables as a facebox
                facebox = {
                    'filename': img_path,
                    'x': face_x,
                    'y': face_y,
                    'w': face_w,
                    'h': face_h
                }
                faceboxes.append(facebox)

    return faceboxes

def get_size_average(faceboxes: list[dict[str, any]]) -> tuple[int, int]:
    """
    Calculate the average width and height of faceboxes.

    Parameters:
    - faceboxes (List[Dict[str, any]]): List of dictionaries representing faceboxes.

    Returns:
    - Tuple[int, int]: Average width and height.
    """
    total_width = 0
    total_height = 0

    # Calculate the sum of width and height for all faceboxes
    for facebox in faceboxes:
        total_width += int(facebox['w'])
        total_height += int(facebox['h'])

    # Calculate the average width and height
    average_width = round(total_width / len(faceboxes))
    average_height = round(total_height / len(faceboxes))

    return average_width, average_height

def main(annotations_path, output_folder):
    faceboxes = read_faceboxes(annotations_path)
    average_w, average_h = get_size_average(faceboxes)

    # Iterate through faceboxes
    for i, facebox in enumerate(faceboxes):
        # Read the image
        img = cv2.imread(facebox['filename'])

        # Extract face box coordinates
        x, y, w, h = int(facebox['x']), int(facebox['y']), int(facebox['w']), int(facebox['h'])

        # Get the differences
        difference_w = average_w - w
        difference_h = average_h - h

        # Get the relative new start coords
        new_x = x - round(difference_w / 2)
        new_y = y - round(difference_h / 2)

        # Check if the resized box is possible
        if new_y + average_h > len(img) or new_x + average_w > len(img[0]):
            continue

        # Crop the region of interest (ROI)
        roi = img[new_y:new_y + average_h, new_x:new_x + average_w]

        # Save the cropped face to the output folder
        output_path = os.path.join(output_folder, f"eigenface_input_{i}.jpg")
        cv2.imwrite(output_path, roi)
        print(f"Cropped Face: {output_path}")

if __name__ == "__main__":
    # Create argument parser
    parser = argparse.ArgumentParser(description="Process faceboxes and save eigenface images.")
    
    # Add arguments
    parser.add_argument("--annotations", type=str, help="Name of the annotations file.")
    parser.add_argument("--output", type=str, help="Folder to save the eigenfaces to.")

    # Parse arguments
    args = parser.parse_args()

    # Check if the annotations file exists
    annotations = args.annotations
    if not os.path.isfile(annotations):
        print(f"Error: Annotations file '{annotations}' does not exist. Please provide a valid file.")
        sys.exit(1)

    # Manage output folder
    output_folder = args.output
    if output_folder == None:
        output_folder = STD_OUTPUT_FOLDER
    os.makedirs(output_folder, exist_ok=True)
    
    # Call the main
    main(annotations, output_folder)
