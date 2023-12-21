import os
import sys
import csv
import argparse

# Standard Constants
STD_INPUT_FOLDER = "input"

def main(folder_path):
    # Get a list of filenames in the specified folder
    files = os.listdir(folder_path)

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
    