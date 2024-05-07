import torch
import random
import numpy as np
from PIL import Image
import torchvision.transforms as transforms


# Convert ndarrays in sample to Tensors.
class ToTensor(object):

    def __call__(self, sample):
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        img1 = sample['image'][0]
        img2 = sample['image'][1]
        mask = sample['label']
        img1 = np.array(img1).astype(np.float32).transpose((2, 0, 1)) # from numpy img dimensions to torch img dimensions
        img2 = np.array(img2).astype(np.float32).transpose((2, 0, 1)) # from numpy img dimensions to torch img dimensions
        mask = np.array(mask).astype(np.float32) / 255.0 # Make the mask [0,1]

        img1 = torch.from_numpy(img1).float() # convert numpy array to torch tensor
        img2 = torch.from_numpy(img2).float() # convert numpy array to torch tensor
        mask = torch.from_numpy(mask).float() # convert numpy array to torch tensor

        return {'image': (img1, img2),
                'label': mask}

# Applies a horizontal flip with a probability of 0.5
class RandomHorizontalFlip(object):
    def __call__(self, sample):
        # img1, img2, mask = sample['image'], sample['label']
        img1 = sample['image'][0]
        img2 = sample['image'][1]
        mask = sample['label']
        # Random horizontal flip
        if random.random() < 0.5:
            # Flip the 3 images
            img1 = img1.transpose(Image.FLIP_LEFT_RIGHT)
            img2 = img2.transpose(Image.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.FLIP_LEFT_RIGHT)

        return {'image': (img1, img2),
                'label': mask}

# Applies a vertical flip with a probability of 0.5
class RandomVerticalFlip(object):
    def __call__(self, sample):
        # img1, img2, mask = sample['image'], sample['label']
        img1 = sample['image'][0]
        img2 = sample['image'][1]
        mask = sample['label']
        # Random vertical flip
        if random.random() < 0.5:
            # Flip the 3 images
            img1 = img1.transpose(Image.FLIP_TOP_BOTTOM)
            img2 = img2.transpose(Image.FLIP_TOP_BOTTOM)
            mask = mask.transpose(Image.FLIP_TOP_BOTTOM)

        return {'image': (img1, img2),
                'label': mask}

# Applies a random rotation with a probability of 0.75 (90, 180, 270 degrees)
class RandomFixRotate(object):
    def __init__(self):
        self.degree = [Image.ROTATE_90, Image.ROTATE_180, Image.ROTATE_270] # 90, 180, 270 degrees (to choose from)

    def __call__(self, sample):
        # img1, img2, mask = sample['image'], sample['label']
        img1 = sample['image'][0]
        img2 = sample['image'][1]
        mask = sample['label']
        # Random rotation
        if random.random() < 0.75:
            rotate_degree = random.choice(self.degree) # Choose a random degree
            # Rotate the 3 images
            img1 = img1.transpose(rotate_degree)
            img2 = img2.transpose(rotate_degree)
            mask = mask.transpose(rotate_degree)

        return {'image': (img1, img2),
                'label': mask}

# Define the transformations for training and testing
train_transforms = transforms.Compose([ # Composes the three transformations
            RandomHorizontalFlip(),
            RandomVerticalFlip(),
            RandomFixRotate(),
            ToTensor()])
# Define the transformations for testing
test_transforms = transforms.Compose([ToTensor()]) # Just to tensor for testing