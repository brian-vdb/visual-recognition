import cv2
import os

def convert_to_grayscale(input_folder, output_folder):
    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

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
            print(f"Converted: {file}")
        else:
            print(f"Error reading image: {file}")

if __name__ == "__main__":
    # Specify the input and output folders
    input_folder = "img"
    output_folder = "img_grayscale"

    # Call the function to convert color images to grayscale
    convert_to_grayscale(input_folder, output_folder)
