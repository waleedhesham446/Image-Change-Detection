# Script to divide the data into training and validation sets
# 80% training and 20% validation

import os
import shutil
import random

def divide(source_folder, train_folder, val_folder, split=0.2):
        # Constant Seed
        random.seed(27)    
        
        # Create the destination folders if they do not exist
        if not os.path.exists(train_folder):
            os.makedirs(train_folder)
        if not os.path.exists(val_folder):
            os.makedirs(val_folder)
            
        # Loop through the files in the source folder
        validation_indices = random.sample(range(len(os.listdir(source_folder))), int(split* len(os.listdir(source_folder))))
        for index, file_name in enumerate(os.listdir(source_folder)):
            # Get the full path of the file
            source_file = os.path.join(source_folder, file_name)
            # Get the full path of the destination file
            if index in validation_indices:
                destination_file = os.path.join(val_folder, file_name)
            else:
                destination_file = os.path.join(train_folder, file_name)
            # Copy the file to the destination folder
            shutil.copy(source_file, destination_file)

divide('./trainval - Copy/A', './trainval - Copy/train/A', './trainval - Copy/val/A')
divide('./trainval - Copy/B', './trainval - Copy/train/B', './trainval - Copy/val/B')
divide('./trainval - Copy/OUT', './trainval - Copy/train/OUT', './trainval - Copy/val/OUT')