import torch
import torch.nn as nn
from torchmetrics.classification import MulticlassCalibrationError, BinaryCalibrationError
from torchmetrics import Metric

class CECE(Metric):
    def __init__(self, num_classes=1, n_bins=1, norm='max') -> None:
        super(CECE, self).__init__(dist_sync_on_step=False)
        self.num_classes = num_classes

        # Use BinaryCalibrationError for each class since each label is treated as a binary task
        self.calibration_error = BinaryCalibrationError(n_bins=n_bins, norm=norm)

        # Add state variables for storing cumulative results
        self.add_state("ece_sum", default=torch.zeros(num_classes), dist_reduce_fx='sum')
        self.add_state("ece_squared_sum", default=torch.zeros(num_classes), dist_reduce_fx='sum')
        self.add_state("total", default=torch.zeros(1), dist_reduce_fx='sum')

    def update(self, logits, labels):
        # Ensure logits and labels are on the correct device by using self.device
        logits, labels = logits.to(self.device), labels.to(self.device)
        
        # Process each label/class independently as a binary classification task
        if logits.dim() == 1: 
            ece = self.calibration_error(logits, labels)
            self.ece_sum[0] += ece
            self.ece_squared_sum[0] += ece ** 2
        else :
            for cls in range(self.num_classes):
                logit_col = logits[:, cls]  # Logits for the current class
                label_col = labels[:, cls]  # Labels for the current class

                # Compute calibration error for the current class
                ece = self.calibration_error(logit_col, label_col)
                self.ece_sum[cls] += ece
                self.ece_squared_sum[cls] += ece ** 2

        # Increment the total number of samples processed
        self.total += 1

    def compute(self):
        # Compute CECE for multi-label: sum of squared errors / sum of errors for each label
        numerator = torch.sum(self.ece_squared_sum)
        denominator = torch.sum(self.ece_sum)
        cece = numerator / denominator if denominator > 0 else torch.zeros(1, device=self.device)
        return cece
