import cv2
import os

def adjust_brightness_grayscale(image, alpha):
    # Adjust image brightness for grayscale image
    adjusted_image = cv2.convertScaleAbs(image, alpha=alpha, beta=0)
    return adjusted_image

def create_bright_dark_images(input_folder, output_folder):
    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # List all files in the input folder
    files = os.listdir(input_folder)

    for file in files:
        # Construct the file paths
        input_path = os.path.join(input_folder, file)

        # Read the grayscale image
        grayscale_image = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)

        if grayscale_image is not None:
            # Create brighter and darker versions of the image
            brighter_image = adjust_brightness_grayscale(grayscale_image, alpha=1.5)
            darker_image = adjust_brightness_grayscale(grayscale_image, alpha=0.5)

            # Remove spaces from the filename
            filename_without_spaces = file.replace(" ", "_")

            # Save the brighter and darker images
            cv2.imwrite(os.path.join(output_folder, f"bright_{filename_without_spaces}"), brighter_image)
            cv2.imwrite(os.path.join(output_folder, f"dark_{filename_without_spaces}"), darker_image)

            print(f"Processed: {file}")
        else:
            print(f"Error reading image: {file}")

if __name__ == "__main__":
    # Specify the input and output folders
    input_folder = "img"
    output_folder = "img_brightness"

    # Call the function to create brighter and darker images
    create_bright_dark_images(input_folder, output_folder)
