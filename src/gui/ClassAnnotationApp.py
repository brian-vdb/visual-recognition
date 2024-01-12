import csv
import tkinter as tk
from PIL import Image, ImageTk

class ClassAnnotationApp:
    """An application for annotating images with class labels."""

    def __init__(self, master, annotations_path):
        """
        Initialize the ImageApp.

        Parameters:
        - master: The master window.
        - annotations_path: Path to the CSV file containing image annotations.
        """
        self.annotations_path = annotations_path
        self.data = self.read_csv(self.annotations_path)
        self.index = 0
        self.master = master

        # Class dictionary to store class labels
        self.current_class = -1
        self.class_dictionary = {}

        # Create a label to display the image
        self.image_label = tk.Label(master)
        self.image_label.pack()

        # Entry widget for creating a new class
        self.new_entry = tk.Entry(master)
        self.new_entry.pack(side=tk.LEFT, padx=5)

        # Button to create a new class
        self.new_button = tk.Button(master, text="Create", command=self.create_class)
        self.new_button.pack(side=tk.LEFT, padx=5)

        # Button to quit the application
        self.quit_button = tk.Button(master, text="Quit", command=self.save_and_quit)
        self.quit_button.pack(side=tk.LEFT, padx=5)

        # Button to delete the current entry
        self.delete_button = tk.Button(master, text="Delete", command=self.delete_entry, width=30)
        self.delete_button.pack(side=tk.TOP, padx=5, pady=5)

        # Display the initial image
        self.show_image()

    def read_csv(self, annotations: str) -> list[dict[str, any]]:
        """
        Read data from a CSV file and return a list of dictionaries.

        Parameters:
        - annotations: Path to the CSV file.

        Returns:
        A list of dictionaries containing image annotations.
        """
        data = []

        # Open the annotations file
        with open(annotations, 'r') as csv_file:
            csv_reader = csv.DictReader(csv_file)

            # Save every entry as a dictionary
            for row in csv_reader:
                entry = {}
                for key, value in row.items():
                    entry[key] = value
                data.append(entry)

        # Return the data
        return data

    def write_csv(self, data: list[dict[str, any]], annotations: str) -> None:
        """Write the data to a CSV file."""
        with open(self.annotations_path, 'w', newline='') as csv_file:
            fieldnames = data[0].keys() if data else []  # Use keys from the first entry as fieldnames
            csv_writer = csv.DictWriter(csv_file, fieldnames=fieldnames)

            # Write the header
            csv_writer.writeheader()

            # Write the data
            csv_writer.writerows(data)

    def save_and_quit(self) -> None:
        """Save the annotations to the CSV file and quit the application."""
        self.write_csv(self.data, self.annotations_path)
        self.master.quit()

    def show_image(self) -> int:
        """
        Display the next image in the list.

        Returns:
        0 if successful, -1 if the last image was reached.
        """
        if self.index >= len(self.data):
            # Return an error if the last image was reached
            self.save_and_quit()
            return -1

        # Display the next image
        image_path = self.data[self.index]['Filename']
        image = Image.open(image_path)
        image.thumbnail((64, 64))
        photo = ImageTk.PhotoImage(image)
        self.image_label.config(image=photo)
        self.image_label.image = photo

        # Pack the label to the right, fill the available space, and expand
        self.image_label.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        return 0

    def insert_and_continue(self, label: int) -> int:
        """
        Label the current image and move to the next one.

        Parameters:
        - label: The class label to assign to the current image.

        Returns:
        0 if successful, -1 if the last image was reached.
        """
        if self.index >= len(self.data):
            # Return an error if the last image was reached
            self.save_and_quit()
            return -1

        # Label the data
        self.data[self.index]['Class'] = label

        # Continue to the next image
        self.index += 1
        return self.show_image()

    def delete_and_continue(self) -> int:
        """
        Delete the current entry and move to the next one.

        Returns:
        0 if successful, -1 if the last image was reached.
        """
        if self.index >= len(self.data):
            # Return an error if the last image was reached
            self.save_and_quit()
            return -1

        # Remove the current image data at the current index
        self.data.pop(self.index)

        # Continue to the next image
        return self.show_image()

    def create_class(self) -> int:
        """
        Create a new class and assign it to the current image.

        Returns:
        0 if successful, -1 if the class already exists or the entry is empty.
        """
        # Get the value from the Entry widget
        new_class = self.new_entry.get()
        if any(new_class == value for value in self.class_dictionary.values()) or new_class == '':
            # Return an error if the class already exists or the entry is empty
            return -1

        # Define the class
        self.current_class += 1
        self.class_dictionary[self.current_class] = new_class

        # Add a button for the newly created class
        self.create_class_button(self.current_class)

        # Handle the labeling
        return self.insert_and_continue(self.current_class)

    def delete_entry(self):
        """Delete the current entry and move to the next one."""
        return self.delete_and_continue()

    def assign_to_existing_class(self, class_id: int) -> int:
        """
        Assign the current image to the selected existing class.

        Parameters:
        - class_id: The ID of the existing class.

        Returns:
        0 if successful, -1 if the last image was reached.
        """
        return self.insert_and_continue(class_id)

    def create_class_button(self, class_id):
        """
        Dynamically create a button for the newly created class.

        Parameters:
        - class_id: The ID of the newly created class.
        """
        class_name = self.class_dictionary[class_id]
        button = tk.Button(self.master, text=f"Assign to {class_name}", command=lambda id=class_id: self.assign_to_existing_class(id), width=30)
        button.pack(side=tk.TOP, padx=5, pady=5)

def run_class_annotation_app(annotations_path: str) -> None:
    # Create the main Tkinter window
    root = tk.Tk()
    root.title("Class Annotation App")

    # Create the ImageAnnotationApp instance
    app = ClassAnnotationApp(root, annotations_path)

    # Run the Tkinter event loop
    root.mainloop()
