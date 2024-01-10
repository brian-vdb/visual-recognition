import os
import sys
import csv
import argparse
import re

# Standard Constants
STD_INPUT_FOLDER = "input"
IMAGE_EXTENSIONS = ['.jpg', '.jpeg', '.png']

def main(folder_path):
    # Get a list of filenames in the specified folder
    files = [f for f in os.listdir(folder_path) if f.lower().endswith(tuple(IMAGE_EXTENSIONS))]
    if len(files) == 0:
        print(f'Error: No images found in the input folder. Please provide a correct image base')
        sys.exit(1)

    # Sort files based on the numerical part of the filename
    files.sort(key=lambda x: [int(c) if c.isdigit() else c for c in re.split('(\d+)', x)])

    # Create or overwrite the CSV file
    with open(os.path.join(folder_path, "eigenfaces.csv"), 'w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)

        # Write header to CSV file
        csv_writer.writerow(['Filename', 'Class'])

        # Write filenames with empty second values to the CSV file
        for file_name in files:
            csv_writer.writerow([os.path.join(folder_path, file_name), ''])

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="List images in a .csv file.")

    # Add arguments
    parser.add_argument("--folder", type=str, help="Folder with images to use as eigenfaces")

    # Parse arguments
    args = parser.parse_args()

    # Manage input folder
    input_folder = args.folder
    if input_folder == None and os.path.exists(STD_INPUT_FOLDER):
        input_folder = STD_INPUT_FOLDER
    elif not os.path.exists(STD_INPUT_FOLDER):
        print(f"Error: Folder not provided and {STD_INPUT_FOLDER} does not exist. Please provide a valid folder.")
        sys.exit(1)
    
    # Call the main
    main(input_folder)
