import torch
from sklearn.metrics import precision_recall_fscore_support as prfs
from utils.args import parse_args
from utils.utilities import (get_loaders)
from Siam_Ecam import Siam_Ecam # Model
from utils.loss_fns import hybrid_loss # Loss function
import os
import json
from tqdm import tqdm
import random
import numpy as np


if __name__ == '__main__':
    """
    Initialize Parser and define arguments
    """
    parser, metadata = parse_args()
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
    # model = torch.load('./checkpoints/checkpoint_epoch_16.pt').to(dev)

    # Load from checkpoint or original
    optimizer = torch.optim.AdamW(model.parameters(), lr=getattr(opt, "validation_metrics", {}).get("learning_rate", opt.learning_rate)) 
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=8, gamma=0.5)

    # When resuming training, update the scheduler to match the current step
    for i in range(getattr(opt, "steps", 0)):
        print("updating scheduler: ", i, "th step.")
        scheduler.step() # Update scheduler to match the current step
         
    # Initilization
    best_metrics = {'f1scores': -1, 'recalls': -1, 'precisions': -1, 'losses': float('inf'), 'jaccard': float('inf')}
    
    steps = getattr(opt, "steps", 0)  # Get the current step from the metadata
    for epoch in range(opt.epochs):
        train_metrics = {'losses': [], 'corrects': [], 'precisions': [],
                        'recalls': [], 'f1scores': [], 'learning_rate': [], 'jaccard': [],
                        } # Metrics for training
        val_metrics = {'losses': [], 'corrects': [], 'precisions': [],
                        'recalls': [], 'f1scores': [], 'learning_rate': [], 'jaccard': [],
                        } # Metrics for validation

        # Training
        model.train()
        batch_iter = 0 # Batch Start
        tbar = tqdm(train_loader)
        for batch_img1, batch_img2, labels in tbar:
            tbar.set_description("epoch {} info ".format(epoch) + str(batch_iter) + " - " + str(batch_iter+opt.batch_size))
            batch_iter = batch_iter+opt.batch_size # Update batch start
            
            # Get images and labels
            batch_img1 = batch_img1.float().to(dev)
            batch_img2 = batch_img2.float().to(dev)
            labels = labels.long().to(dev)

            # Zero the gradient
            optimizer.zero_grad()

            # Get model predictions, calculate loss, backprop
            preds = model(batch_img1, batch_img2)

            loss, iou = hybrid_loss(preds, labels) # Loss function
            loss = loss
            loss.backward() # Backpropagation
            optimizer.step() # Update the weights

            preds = preds[-1]
            _, preds = torch.max(preds, 1)

            # Calculate batch metrics
            corrects = (100 *
                        (preds.squeeze().byte() == labels.squeeze().byte()).sum() /
                        (labels.size()[0] * (opt.patch_size**2)))

            train_report = prfs(labels.data.cpu().numpy().flatten(),
                                preds.data.cpu().numpy().flatten(),
                                average='binary',
                                zero_division=0,
                                pos_label=1)
            
            # Assign Metrics of current batch to the epoch metrics
            train_metrics['losses'].append(loss.item())
            train_metrics['jaccard'].append(iou.item())
            train_metrics['corrects'].append(corrects.item())
            train_metrics['precisions'].append(train_report[0])
            train_metrics['recalls'].append(train_report[1])
            train_metrics['f1scores'].append(train_report[2])
            train_metrics['learning_rate'].append(scheduler.get_last_lr())

            
            # log the batch mean metrics
            mean_train_metrics  = {k: np.mean(v) for k, v in train_metrics.items()}


            # clear batch variables from memory
            del batch_img1, batch_img2, labels

        scheduler.step()
        steps = steps % 8 + 1 # ciurcular increment
        print(f'Epoch{epoch} training finished. {mean_train_metrics}')

        # Evaluationq
        model.eval()
        with torch.no_grad():
            for batch_img1, batch_img2, labels in val_loader:
                # Set variables for training
                batch_img1 = batch_img1.float().to(dev)
                batch_img2 = batch_img2.float().to(dev)
                labels = labels.long().to(dev)

                # Get predictions and calculate loss
                preds = model(batch_img1, batch_img2)

                loss, iou = hybrid_loss(preds, labels) # Loss function

                preds = preds[-1]
                _, preds = torch.max(preds, 1)

                # Calculate and log other batch metrics
                corrects = (100 *
                            (preds.squeeze().byte() == labels.squeeze().byte()).sum() /
                            (labels.size()[0] * (opt.patch_size**2)))

                val_report = prfs(labels.data.cpu().numpy().flatten(),
                                    preds.data.cpu().numpy().flatten(),
                                    average='binary',
                                    zero_division=0,
                                    pos_label=1)
                
                # Assign Metrics of current batch to the epoch metrics
                val_metrics['losses'].append(loss.item())
                val_metrics['jaccard'].append(iou.item())
                val_metrics['corrects'].append(corrects.item())
                val_metrics['precisions'].append(val_report[0])
                val_metrics['recalls'].append(val_report[1])
                val_metrics['f1scores'].append(val_report[2])
                val_metrics['learning_rate'].append(scheduler.get_last_lr())

                # log the batch mean metrics
                mean_val_metrics = {k: np.mean(v) for k, v in val_metrics.items()}

                # clear batch variables from memory
                del batch_img1, batch_img2, labels

            print(f'Epoch{epoch} validation finished. {mean_val_metrics}')
            
            # If this epoch is better than the previous best, save the model and log
            if (mean_val_metrics['losses'] < best_metrics['losses'] 
                 or mean_val_metrics['jaccard'] < best_metrics['jaccard']):

                # Insert training and epoch information to metadata dictionary
                metadata['steps'] = steps
                metadata['validation_metrics'] = mean_val_metrics

                # Save model and log
                if not os.path.exists('./checkpoints'):
                    os.mkdir('./checkpoints')
                with open('./checkpoints/metadata_epoch_' + str(epoch) + '.json', 'w') as fout:
                    json.dump(metadata, fout)

                torch.save(model, './checkpoints/checkpoint_epoch_'+str(epoch)+'.pt')

                best_metrics = mean_val_metrics


            print('An epoch finished.')
    print('Done!')
