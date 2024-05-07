import torch.utils.data
from utils.args import parse_args
from utils.utilities import get_test_loaders
from tqdm import tqdm
from sklearn.metrics import confusion_matrix

# Test the model
if __name__ == '__main__':
    parser, metadata = parse_args()
    opt = parser.parse_args()

    dev = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    test_loader = get_test_loaders(opt)

    path = './tmp/checkpoint_epoch_24.pt'   # the path of the model
    model = torch.load(path)

    c_matrix = {'tn': 0, 'fp': 0, 'fn': 0, 'tp': 0}
    model.eval()

    with torch.no_grad():
        tbar = tqdm(test_loader)
        for batch_img1, batch_img2, labels in tbar: # For each batch of images
            # Convert the images to float and move them to the device
            batch_img1 = batch_img1.float().to(dev)
            batch_img2 = batch_img2.float().to(dev)
            labels = labels.long().to(dev) # Convert the labels to long and move them to the device

            # Get the predictions from the model
            preds = model(batch_img1, batch_img2)
            preds = preds[-1] 
            _, preds = torch.max(preds, 1)

            # Calculate the confusion matrix for the batch
            cm = confusion_matrix(labels.data.cpu().numpy().flatten(),
                            preds.data.cpu().numpy().flatten(), labels=[0,1]).ravel()
            
            # Update the confusion matrix (For the entire dataset)
            c_matrix['tn'] += cm[0]
            c_matrix['fp'] += cm[1]
            c_matrix['fn'] += cm[2]  
            c_matrix['tp'] += cm[3]

    tn, fp, fn, tp = c_matrix['tn'], c_matrix['fp'], c_matrix['fn'], c_matrix['tp'] # Just for readability
    P = tp / (tp + fp) # Precision
    R = tp / (tp + fn) # Recall
    F1 = 2 * P * R / (R + P) # F1-Score
    iou = tp / (tp + fp + fn) # Jaccard Index

    print('Precision: {}\nRecall: {}\nF1-Score: {}, \nJaccard Index: {}'.format(P, R, F1, iou))
