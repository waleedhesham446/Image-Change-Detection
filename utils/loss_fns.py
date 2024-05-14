import torch
import torch.utils.data
import torch.nn as nn
import torch.nn.functional as F
import warnings
import numpy as np

warnings.filterwarnings("ignore", category=UserWarning)

# Define the focal loss function (Used here for cross-entropy loss -> gamma=0, alpha=None)
class FocalLoss(nn.Module):
    def __init__(self, gamma=0, alpha=None):
        super(FocalLoss, self).__init__()
        # These arguments are just to make the loss function more flexible (not necessary for this task)
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha, (float, int)):
            self.alpha = torch.Tensor([alpha, 1-alpha])
        if isinstance(alpha, list):
            self.alpha = torch.Tensor(alpha)

    def forward(self, input, target):
        if input.dim() > 2:
            input = input.view(input.size(0), input.size(1), -1).transpose(1, 2).contiguous().view(-1, input.size(1))

        target = target.view(-1, 1) # Flatten the target
        prob = F.log_softmax(input) # Apply log-softmax to the input
        prob = prob.gather(1, target).view(-1) # Get the probabilities of the target class

        loss = -prob # Calculate the loss

        return loss.mean() # Return the mean loss

# Define the dice loss function
def dice_loss(logits, true, eps=1e-7):
    # Determine the number of classes
    num_classes = logits.shape[1]
    
    # Create one-hot encoded tensors and softmax probabilities
    true_one_hot = torch.eye(num_classes, device=true.device)[true.squeeze(1)]
    true_one_hot = true_one_hot.permute(0, 3, 1, 2).float()
    probabilities = F.softmax(logits, dim=1)
    
    # Ensure the true_one_hot tensor type matches logits
    true_one_hot = true_one_hot.type(logits.type())
    
    # Calculate intersection and cardinality
    dims = (0,) + tuple(range(2, true.ndimension()))
    intersection = torch.sum(probabilities * true_one_hot, dims)
    cardinality = torch.sum(probabilities + true_one_hot, dims)
    
    # Calculate Dice loss
    dice_loss = (2. * intersection / (cardinality + eps)).mean()
    
    return (1 - dice_loss)

# Define the Jaccard loss function
def jaccard_loss(logits, true, eps=1e-7):
    num_classes = logits.shape[1]
    
    true_one_hot = torch.eye(num_classes, device=true.device)[true.squeeze(1)]
    true_one_hot = true_one_hot.permute(0, 3, 1, 2).float()
    probabilities = F.softmax(logits, dim=1)
    
    
    true_one_hot = true_one_hot.type(logits.type())
    
    dims = (0,) + tuple(range(2, true.ndimension()))
    intersection = torch.sum(probabilities * true_one_hot, dims)
    cardinality = torch.sum(probabilities + true_one_hot, dims)
    union = cardinality - intersection
    jaccard_index = (intersection / (union + eps)).mean()
    
    return (1 - jaccard_index)


def jaccard_loss_2(img1, img2):
    intersection = np.logical_and(img1, img2).sum()
    union = np.logical_or(img1, img2).sum()
    
    if union != 0:
        jaccard_index = intersection / union
    else:
        jaccard_index = 1
    
    jaccard_loss = 1 - jaccard_index
    
    return jaccard_loss

# Hybrid loss function that incorporates both cross-entropy and dice loss
def hybrid_loss(predictions, target):
    # Compute loss
    loss = 0
    iou = 0

    print(len(predictions), len(target))
    print(predictions[0].shape, target[0].shape)
    focal = FocalLoss(gamma=0, alpha=None)
    for prediction in predictions:

        bce = focal(prediction, target) # get cross-entropy loss -> More Stable
        dice = dice_loss(prediction, target) # get dice loss -> Can handle class imbalance
        loss += bce + dice # hybrid loss is the sum of the two losses
        
        jaccard = jaccard_loss_2(prediction, target)
        iou += jaccard
        
    return loss, iou # return the hybrid loss

