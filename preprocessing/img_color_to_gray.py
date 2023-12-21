import cv2
import os
import argparse
import sys

# Standard Constants
STD_INPUT_FOLDER = "input"
STD_OUTPUT_FOLDER = "output"

def main(input_folder, output_folder):
    # List all files in the input folder
    files = os.listdir(input_folder)

    for file in files:
        # Construct the file paths
        input_path = os.path.join(input_folder, file)
        output_path = os.path.join(output_folder, file)

        # Read the color image
        color_image = cv2.imread(input_path)

        if color_image is not None:
            # Convert the color image to grayscale
            grayscale_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)

            # Save the grayscale image
            cv2.imwrite(output_path, grayscale_image)
        else:
            print(f"Error reading image: {file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process images and save as grayscale images.")

    # Add arguments
    parser.add_argument("--input", type=str, help="Folder to get the normal images from")
    parser.add_argument("--output", type=str, help="Folder to save the grayscale images to.")

    # Parse arguments
    args = parser.parse_args()

    # Manage input folder
    input_folder = args.input
    if input_folder == None and os.path.exists(STD_INPUT_FOLDER):
        input_folder = STD_INPUT_FOLDER
    elif not os.path.exists(STD_INPUT_FOLDER):
        print(f"Error: Input folder not provided and {STD_INPUT_FOLDER} does not exist. Please provide a valid folder.")
        sys.exit(1)

    # Manage output folder
    output_folder = args.output
    if output_folder == None:
        output_folder = STD_OUTPUT_FOLDER
    os.makedirs(output_folder, exist_ok=True)

    # Call the main
    main(input_folder, output_folder)
