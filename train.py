import torch
from sklearn.metrics import precision_recall_fscore_support as prfs
from utils.parser import get_parser_with_args
from utils.helpers import (get_loaders)
from models.Siam_Ecam import Siam_Ecam # Model
from utils.losses import hybrid_loss # Loss function
import os
import json
from tqdm import tqdm
import random
import numpy as np


if __name__ == '__main__':
    """
    Initialize Parser and define arguments
    """
    parser, metadata = get_parser_with_args()
    opt = parser.parse_args()


    # define paths, load data
    dev = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    def seed_torch(seed):
        random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    seed_torch(seed=777)


    train_loader, val_loader = get_loaders(opt)

    # Load Model, define loss function and optimizer
    model = Siam_Ecam(3, 2).to(dev)

    criterion = hybrid_loss
    optimizer = torch.optim.AdamW(model.parameters(), lr=opt.learning_rate) # Be careful when you adjust learning rate, you can refer to the linear scaling rule
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=8, gamma=0.5)

    # Initilization
    best_metrics = {'cd_f1scores': -1, 'cd_recalls': -1, 'cd_precisions': -1}
    total_step = -1

    for epoch in range(opt.epochs):
        train_metrics = {'cd_losses': [], 'cd_corrects': [], 'cd_precisions': [],
                        'cd_recalls': [], 'cd_f1scores': [], 'learning_rate': [],
                        }
        val_metrics = {'cd_losses': [], 'cd_corrects': [], 'cd_precisions': [],
                        'cd_recalls': [], 'cd_f1scores': [], 'learning_rate': [],
                        }

        # Training
        model.train()
        batch_iter = 0
        tbar = tqdm(train_loader)
        for batch_img1, batch_img2, labels in tbar:
            tbar.set_description("epoch {} info ".format(epoch) + str(batch_iter) + " - " + str(batch_iter+opt.batch_size))
            batch_iter = batch_iter+opt.batch_size
            total_step += 1
            # Set variables for training
            batch_img1 = batch_img1.float().to(dev)
            batch_img2 = batch_img2.float().to(dev)
            labels = labels.long().to(dev)

            # Zero the gradient
            optimizer.zero_grad()

            # Get model predictions, calculate loss, backprop
            cd_preds = model(batch_img1, batch_img2)

            cd_loss = criterion(cd_preds, labels)
            loss = cd_loss
            loss.backward()
            optimizer.step()

            cd_preds = cd_preds[-1]
            _, cd_preds = torch.max(cd_preds, 1)

            # Calculate and log other batch metrics
            cd_corrects = (100 *
                        (cd_preds.squeeze().byte() == labels.squeeze().byte()).sum() /
                        (labels.size()[0] * (opt.patch_size**2)))

            cd_train_report = prfs(labels.data.cpu().numpy().flatten(),
                                cd_preds.data.cpu().numpy().flatten(),
                                average='binary',
                                zero_division=0,
                                pos_label=1)
            
            # Assign Metrics of current batch to the epoch metrics
            train_metrics['cd_losses'].append(cd_loss.item())
            train_metrics['cd_corrects'].append(cd_corrects.item())
            train_metrics['cd_precisions'].append(cd_train_report[0])
            train_metrics['cd_recalls'].append(cd_train_report[1])
            train_metrics['cd_f1scores'].append(cd_train_report[2])
            train_metrics['learning_rate'].append(scheduler.get_last_lr())

            
            # log the batch mean metrics
            mean_train_metrics  = {k: np.mean(v) for k, v in train_metrics.items()}


            # clear batch variables from memory
            del batch_img1, batch_img2, labels

        scheduler.step()

        # Evaluationq
        model.eval()
        with torch.no_grad():
            for batch_img1, batch_img2, labels in val_loader:
                # Set variables for training
                batch_img1 = batch_img1.float().to(dev)
                batch_img2 = batch_img2.float().to(dev)
                labels = labels.long().to(dev)

                # Get predictions and calculate loss
                cd_preds = model(batch_img1, batch_img2)

                cd_loss = criterion(cd_preds, labels)

                cd_preds = cd_preds[-1]
                _, cd_preds = torch.max(cd_preds, 1)

                # Calculate and log other batch metrics
                cd_corrects = (100 *
                            (cd_preds.squeeze().byte() == labels.squeeze().byte()).sum() /
                            (labels.size()[0] * (opt.patch_size**2)))

                cd_val_report = prfs(labels.data.cpu().numpy().flatten(),
                                    cd_preds.data.cpu().numpy().flatten(),
                                    average='binary',
                                    zero_division=0,
                                    pos_label=1)
                
                # Assign Metrics of current batch to the epoch metrics
                val_metrics['cd_losses'].append(cd_loss.item())
                val_metrics['cd_corrects'].append(cd_corrects.item())
                val_metrics['cd_precisions'].append(cd_val_report[0])
                val_metrics['cd_recalls'].append(cd_val_report[1])
                val_metrics['cd_f1scores'].append(cd_val_report[2])
                val_metrics['learning_rate'].append(scheduler.get_last_lr())

                # log the batch mean metrics
                mean_val_metrics = {k: np.mean(v) for k, v in val_metrics.items()}

                # clear batch variables from memory
                del batch_img1, batch_img2, labels


            # If this epoch is better than the previous best, save the model and log
            if ((mean_val_metrics['cd_precisions'] > best_metrics['cd_precisions'])
                    or
                    (mean_val_metrics['cd_recalls'] > best_metrics['cd_recalls'])
                    or
                    (mean_val_metrics['cd_f1scores'] > best_metrics['cd_f1scores'])):

                # Insert training and epoch information to metadata dictionary
                metadata['validation_metrics'] = mean_val_metrics

                # Save model and log
                if not os.path.exists('./tmp'):
                    os.mkdir('./tmp')
                with open('./tmp/metadata_epoch_' + str(epoch) + '.json', 'w') as fout:
                    json.dump(metadata, fout)

                torch.save(model, './tmp/checkpoint_epoch_'+str(epoch)+'.pt')

                # comet.log_asset(upload_metadata_file_path)
                best_metrics = mean_val_metrics


            print('An epoch finished.')
    print('Done!')
