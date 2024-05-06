import logging
import torch
import torch.utils.data
import torch.nn as nn
import numpy as np
from utils.dataloaders import (full_path_loader, full_test_loader, CDDloader)
from utils.losses import hybrid_loss, dice_loss
from models.Models import Siam_NestedUNet_Conc, SNUNet_ECAM
from models.Siam_Ecam import Siam_Ecam
from models.siamunet_dif import SiamUnet_diff
logging.basicConfig(level=logging.INFO)

def initialize_metrics():
    """Generates a dictionary of metrics with metrics as keys
       and empty lists as values

    Returns
    -------
    dict
        a dictionary of metrics

    """
    metrics = {
        'cd_losses': [],
        'cd_corrects': [],
        'cd_precisions': [],
        'cd_recalls': [],
        'cd_f1scores': [],
        'learning_rate': [],
    }

    return metrics



def set_test_metrics(metric_dict, cd_corrects, cd_report):

    metric_dict['cd_corrects'].append(cd_corrects.item())
    metric_dict['cd_precisions'].append(cd_report[0])
    metric_dict['cd_recalls'].append(cd_report[1])
    metric_dict['cd_f1scores'].append(cd_report[2])

    return metric_dict


def get_loaders(opt):


    logging.info('STARTING Dataset Creation')

    train_full_load, val_full_load = full_path_loader(opt.dataset_dir)


    train_dataset = CDDloader(train_full_load, aug=opt.augmentation)
    val_dataset = CDDloader(val_full_load, aug=False)

    logging.info('STARTING Dataloading')

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=opt.batch_size,
                                               shuffle=True,
                                               num_workers=opt.num_workers)
    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=opt.batch_size,
                                             shuffle=False,
                                             num_workers=opt.num_workers)
    return train_loader, val_loader

def get_test_loaders(opt, batch_size=None):

    if not batch_size:
        batch_size = opt.batch_size

    logging.info('STARTING Dataset Creation')

    test_full_load = full_test_loader(opt.dataset_dir)

    test_dataset = CDDloader(test_full_load, aug=False)

    logging.info('STARTING Dataloading')


    test_loader = torch.utils.data.DataLoader(test_dataset,
                                             batch_size=batch_size,
                                             shuffle=False,
                                             num_workers=opt.num_workers)
    return test_loader
