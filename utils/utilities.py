import torch
import torch.utils.data
from utils.dataloaders import (train_val_loader, test_loader, ImageLoader)

# Get the training and validation loaders
def get_loaders(opt):


    print('Loading Data...')

    train_paths, val_paths = train_val_loader(opt.dataset_dir) # Get the training and validation data paths


    train_dataset = ImageLoader(train_paths, aug=opt.augmentation) # Loader for training data
    val_dataset = ImageLoader(val_paths, aug=False) # Loader for validation data

    # Create the data loaders
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=opt.batch_size,
                                               shuffle=True,
                                               num_workers=opt.num_workers)
    # Validation loader
    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=opt.batch_size,
                                             shuffle=False,
                                             num_workers=opt.num_workers)
    return train_loader, val_loader

# Get the test loader
def get_test_loaders(opt, batch_size=None):

    if not batch_size:
        batch_size = opt.batch_size # Get the batch size from the options

    print('Loading Test Data...')

    test_paths = test_loader(opt.dataset_dir) # Get the test data paths

    test_dataset = ImageLoader(test_paths, aug=False) # Loader for test data

    # Create the test loader
    test_data_loader = torch.utils.data.DataLoader(test_dataset,
                                             batch_size=batch_size,
                                             shuffle=False,
                                             num_workers=opt.num_workers)
    return test_data_loader
