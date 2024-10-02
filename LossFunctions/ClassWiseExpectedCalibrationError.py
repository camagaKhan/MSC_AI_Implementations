import torch
import torch.nn as nn
from torchmetrics.classification import MulticlassCalibrationError, BinaryCalibrationError
from torchmetrics import Metric 

# from here: https://caiac.pubpub.org/pub/vd3v9vby/release/1
class CECE (Metric):
    # norm can be: [l1, l2, max]    
    def __init__(self, num_classes = 1, n_bins = 1, norm='max') -> None:
        super(CECE, self).__init__(dist_sync_on_step=False)
        self.num_classes = num_classes
        self.ece_list = []
        
        if self.num_classes == 1 : 
            self.calibration_error = BinaryCalibrationError(n_bins=n_bins, norm=norm)
        elif self.num_classes > 1:
            self.calibration_error = MulticlassCalibrationError(num_classes=self.num_classes, n_bins=n_bins, norm=norm)
        else : 
            # throw an error
            raise ValueError('num_classes must be a positive integer.')
        
        self.add_state("ece_list", default=[], dist_reduce_fx='cat')
        
    def update(self, logits, labels):
        for cls in range(self.num_classes):
            label_col = labels[:, cls] if self.num_classes > 1 else labels # get the column values of the true labels
            self.ece_list.append(self.calibration_error(logits, label_col)) # get the expected calibration loss
    
    '''
      Pass the original logits, not the ones in sigmoid.
    '''        
    def compute(self):     
        ece_tensor = torch.tensor(self.ece_list)
        numerator, denominator = torch.sum(ece_tensor ** 2), torch.sum(ece_tensor)
        cece = numerator / denominator if denominator > 0 else torch.zeros(1)        
        return cece