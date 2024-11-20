import torch
import torch.nn.functional as F


class BCEWithLogitsLossMultitask(torch.nn.Module):
    """ Custom Loss function for multi-task scenario"""
    def __init__(self):
        """Custom implementation of a loss functions for a multi-task scenario
        """
        super().__init__()
        self.bce = torch.nn.BCEWithLogitsLoss(reduction="none")
    
    def forward(self, input, target):
        """Calculate loss for each sample

        Args:
            input (Tensor): Logit tensor of dimensions (batch, annotators)
            target (Tensor): Tensor of true label for each annotator

        Returns:
            Tensor: For each sample, we calculate the loss for each annotator with BCE, 
            then we calculate the loss as sum of the loss of each annotator
        """

        return  torch.sum(self.bce(input, target), dim=1)

class CEWithLogitsLoss(torch.nn.Module):
    def __init__(self):
        """Custom implementation of CE function
        """
        super().__init__()
        self.ce = torch.nn.CrossEntropyLoss(reduction='none')
    
    def forward(self, input, target):
        """Custom CrossEntropyLoss

        Args:
            input (Tensor): Logits prediction
            target (Tensor): Contains probabilities for class = 1 for a given sample.

        Returns:
            Tensor: CrossEntropy
        """
        target = torch.stack((1 - target, target), dim=1)
        return self.ce(input, target)


class BCEWithLogitsLoss(torch.nn.Module):
    """BCEWithLogitsLoss of PyTorch. Implemeted here for a more clean code
    """
    def __init__(self):
        super().__init__()
        self.bce =  torch.nn.BCEWithLogitsLoss(reduction='none')
    
    def forward(self, input, target):
        input.squeeze_(dim=1)
        return self.bce(input, target)


class KLDivLoss(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.kldiv = torch.nn.KLDivLoss(reduction="batchmean")
    
    def forward(self, input, target):
        target.unsqueeze_(1)
        target = torch.cat((1 - target, target), dim=1)

        input = F.log_softmax(input, dim=1)
        
        return self.kldiv(input, target)