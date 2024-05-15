# Script to combine the contents of 3 folders into a single folder

import os
import shutil

def combine(source_folders, destination_folder):

    # Create the destination folder if it does not exist
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)
        
    # Loop through the source folders
    for source_folder in source_folders:
        # Loop through the files in the source folder
        for file_name in os.listdir(source_folder):
            # Get the full path of the file
            source_file = os.path.join(source_folder, file_name)
            # Get the full path of the destination file
            destination_file = os.path.join(destination_folder, file_name)
            # Copy the file to the destination folder
            shutil.copy(source_file, destination_file)
       
for dir in ['A', 'B', 'OUT']:     
    # Define the source folders and the destination folder
    source_folders = [f'./trainval - Copy/train/{dir}', f'./trainval - Copy/test/{dir}', f'./trainval - Copy/val/{dir}']
    destination_folder = f'./trainval - Copy/{dir}'

    # Call the combine function
    combine(source_folders, destination_folder)
    
    print(f'Finished combining {dir} folders')
