import torch 
import torch.nn as nn

class FocalLoss(nn.Module) : 
    def __init__(self, alpha=None, gamma=2., pos_weights = []) -> None:
        super(FocalLoss, self).__init__()       
        self.alpha = alpha 
        self.gamma = gamma
        
        if len(pos_weights) > 0:         
            self.criterion = nn.BCEWithLogitsLoss(reduction='none', pos_weight=pos_weights)
        else:
            self.criterion = nn.BCEWithLogitsLoss(reduction='none')
        
    def forward(self, inputs, targets):
        loss = self.criterion(inputs, targets)
        targets = targets.float()
        pt = torch.exp(-loss)    
        
        alpha_tensor = self.alpha * targets + (1-self.alpha) * (1-targets)
        
        focal_loss = alpha_tensor * ((1 - pt) ** self.gamma) * loss
        return focal_loss.mean() # <-- to update this
