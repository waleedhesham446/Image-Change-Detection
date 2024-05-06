from utils.parser import get_parser_with_args
import torch
import torch.utils.data
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

class FocalLoss(nn.Module):
    def __init__(self, gamma=0, alpha=None):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha, (float, int)):
            self.alpha = torch.Tensor([alpha, 1-alpha])
        if isinstance(alpha, list):
            self.alpha = torch.Tensor(alpha)

    def forward(self, input, target):
        if input.dim() > 2:
            input = input.view(input.size(0), input.size(1), -1).transpose(1, 2).contiguous().view(-1, input.size(1))

        target = target.view(-1, 1)
        prob = F.log_softmax(input)
        prob = prob.gather(1, target).view(-1)

        loss = -prob

        return loss.mean()

def dice_loss(logits, true, eps=1e-7):
    """Computes the Sørensen–Dice loss.
    Note that PyTorch optimizers minimize a loss. In this
    case, we would like to maximize the dice loss so we
    return the negated dice loss.
    Args:
        true: a tensor of shape [B, 1, H, W].
        logits: a tensor of shape [B, C, H, W]. Corresponds to
            the raw output or logits of the model.
        eps: added to the denominator for numerical stability.
    Returns:
        dice_loss: the Sørensen–Dice loss.
    """
    num_classes = logits.shape[1]
    if num_classes == 1:
        true_one_hot = torch.eye(num_classes + 1)[true.squeeze(1)]
        true_one_hot = true_one_hot.permute(0, 3, 1, 2).float()
        true_one_hot_f = true_one_hot[:, 0:1, :, :]
        true_one_hot_s = true_one_hot[:, 1:2, :, :]
        true_one_hot = torch.cat([true_one_hot_s, true_one_hot_f], dim=1)
        pos_prob = torch.sigmoid(logits)
        neg_prob = 1 - pos_prob
        probas = torch.cat([pos_prob, neg_prob], dim=1)
    else:
        true_one_hot = torch.eye(num_classes, device=true.device)[true.squeeze(1)]
        true_one_hot = true_one_hot.permute(0, 3, 1, 2).float()
        probas = F.softmax(logits, dim=1)
    true_one_hot = true_one_hot.type(logits.type())
    dims = (0,) + tuple(range(2, true.ndimension()))
    intersection = torch.sum(probas * true_one_hot, dims)
    cardinality = torch.sum(probas + true_one_hot, dims)
    dice_loss = (2. * intersection / (cardinality + eps)).mean()
    return (1 - dice_loss)

parser, metadata = get_parser_with_args()
opt = parser.parse_args()

def hybrid_loss(predictions, target):
    """Calculating the loss"""
    loss = 0

    # gamma=0, alpha=None --> CE
    focal = FocalLoss(gamma=0, alpha=None)
    for prediction in predictions:

        bce = focal(prediction, target)
        dice = dice_loss(prediction, target)
        loss += bce + dice

    return loss

