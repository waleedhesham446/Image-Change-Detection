import os
import torch.utils.data as data
from PIL import Image
from utils import augmentation as ag # Import the augmentation file



# Get Img paths (A, B), and label paths (full path)
def train_val_loader(data_dir):
    # Get all training image names
    train_data = [image for image in os.listdir(data_dir + 'train/A/') if not
    image.startswith('.')]
    train_data.sort()

    # Get all validation image names
    val_data = [image for image in os.listdir(data_dir + 'val/A/') if not
    image.startswith('.')]
    val_data.sort()

    train_label_paths = [] # Full path of the label images
    val_label_paths = [] # Full path of the label images
    # For each image name
    for img in train_data:
        train_label_paths.append(data_dir + 'train/OUT/' + img) # Append the full path of the label image
    for img in val_data: 
        val_label_paths.append(data_dir + 'val/OUT/' + img) # Append the full path of the label image


    train_data_path = []
    val_data_path = []

    for img in train_data:
        train_data_path.append([data_dir + 'train/', img])  # Append the partial path of the image (Missing A,B)
    for img in val_data:
        val_data_path.append([data_dir + 'val/', img]) # Append the partial path of the image (Missing A,B)

    train_dataset = {} # Dictionary to store the training data
    val_dataset = {} # Dictionary to store the validation data
    for img_name in range(len(train_data)): # For each image in the training data
        # Add the image and label to the train dictionary
        train_dataset[img_name] = {'image': train_data_path[img_name],
                         'label': train_label_paths[img_name]}
    for img_name in range(len(val_data)):
        # Add the image and label to the val dictionary
        val_dataset[img_name] = {'image': val_data_path[img_name],
                         'label': val_label_paths[img_name]}


    return train_dataset, val_dataset # Return the training and validation data (image paths and label paths)


# Get testing img paths (A, B), and label paths (full path)
def test_loader(data_dir):
    # Get all testing image names
    test_data = [image for image in os.listdir(data_dir + 'test/A/') if not
                    image.startswith('.')]
    test_data.sort()

    test_label_paths = [] # Full path of the label images
    for img in test_data: # For each image name
        test_label_paths.append(data_dir + 'test/OUT/' + img) # Append the full path of the label image

    test_data_path = [] # Partial path of the image (Missing A,B)
    for img in test_data:
        test_data_path.append([data_dir + 'test/', img]) # Append the partial path of the image (Missing A,B)

    test_dataset = {} # Dictionary to store the testing data
    for img_name in range(len(test_data)): # For each image in the testing data
        # Add the image and label to the test dictionary
        test_dataset[img_name] = {'image': test_data_path[img_name],
                           'label': test_label_paths[img_name]}

    return test_dataset # Return the testing data (image paths and label paths)

# For getting images from paths
def images_loader(img_path, label_path, aug):
    dir = img_path[0] # Main directory
    name = img_path[1] # Image name

    img1 = Image.open(dir + 'A/' + name) # Access images from directory A
    img2 = Image.open(dir + 'B/' + name) # Access images from directory B
    label = Image.open(label_path) # Access label images
    sample = {'image': (img1, img2), 'label': label} # Store the images and labels in a dictionary

    if aug: 
        sample = ag.train_transforms(sample) # Apply the training augnentations
    else:
        sample = ag.test_transforms(sample) # Convert the images to tensors (no augmentation) -> Testing and Validation

    return sample['image'][0], sample['image'][1], sample['label']

# class for loading images using the image loader function
class ImageLoader(data.Dataset):

    def __init__(self, path_load, aug=False):

        self.path_load = path_load # img paths and label paths
        self.loader = images_loader # image loader function
        self.aug = aug

    def __getitem__(self, index):
        # Get the image and label paths
        img_path, label_path = self.path_load[index]['image'], self.path_load[index]['label']
        # Load the images and labels using the loader function and return them
        return self.loader(img_path,
                           label_path,
                           self.aug)

    def __len__(self):
        return len(self.path_load)
