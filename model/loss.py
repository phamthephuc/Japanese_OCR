import torch.nn.functional as F
import torch.nn as nn

def nll_loss(output, target):
    return F.nll_loss(output, target)

class CTCLoss(nn.CTCLoss):
    def __init__(self):
        super().__init__()