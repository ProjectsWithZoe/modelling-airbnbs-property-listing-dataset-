import os
import shutil
from PIL import Image

import os
from PIL import Image

# Define the source and destination folders
source_folder = 'airbnb-property-listings/images'
destination_folder = 'airbnb-property-listings/processed_images'

def resize_images(source_folder, destination_folder):
    # Loop through each folder in the source directory
    for folder_name in os.listdir(source_folder):
        # Check if the current file is a directory
        if os.path.isdir(os.path.join(source_folder, folder_name)):
            # Define the source and destination paths for this folder
            current_source_path = os.path.join(source_folder, folder_name)
            current_destination_path = os.path.join(destination_folder, folder_name)

            # Create the destination folder if it doesn't exist
            if not os.path.exists(current_destination_path):
                os.makedirs(current_destination_path)

            # Initialize the minimum height variable to a large value
            min_height = float('inf')

            # Loop through each file in the current folder
            for file_name in os.listdir(current_source_path):
                # Check if the file is a PNG image
                if file_name.endswith('.png'):
                    # Open the image file
                    image_path = os.path.join(current_source_path, file_name)
                    image = Image.open(image_path)

                    # Get the height of the image
                    height = image.size[1]

                    # Update the minimum height variable if necessary
                    if height < min_height:
                        min_height = height

            # Loop through each file in the current folder again and resize the images
            for file_name in os.listdir(current_source_path):
                # Check if the file is a PNG image
                if file_name.endswith('.png'):
                    # Open the image file
                    image_path = os.path.join(current_source_path, file_name)
                    image = Image.open(image_path)

                    # Resize the image to the minimum height
                    width, height = image.size
                    new_width = int(width * (min_height / height))
                    new_size = (new_width, min_height)
                    resized_image = image.resize(new_size)

                    # Save the resized image to the destination folder with a new name
                    new_file_name = os.path.splitext(file_name)[0] + '_resized.png'
                    new_file_path = os.path.join(current_destination_path, new_file_name)
                    resized_image.save(new_file_path)
    for folder_name in os.listdir(destination_folder):
        count = 0
        if os.path.isdir(os.path.join(destination_folder, folder_name)):
            current_folder = os.path.join(destination_folder,folder_name)             
            for file_name in os.listdir(current_folder):
               file_path = os.path.join(current_folder, file_name)
               image = Image.open(file_path)
               if image.mode!= 'RGB':
                   os.remove(file_path) 
                   count+=1
                   print(count)

if __name__ == '__main__':
    resize_images(source_folder, destination_folder)


