import torch 
import torch.nn as nn
from pytorch_lightning import LightningModule
from tqdm import tqdm
import sys
sys.path.append('./')
from LossFunctions.ClassWiseExpectedCalibrationError import CECE
from LossFunctions.FocalLoss import FocalLoss


# based on this implementation: https://github.com/gpleiss/temperature_scaling/blob/master/temperature_scaling.py

class TemperatureScaling(LightningModule):
    
    def __init__(self, model, temperature, n_bins = 10, lr = 0.1, num_labels= 6, alpha = .25, gamma = 2, useFocalLoss = False):
        super(TemperatureScaling, self).__init__()
        self.model = model
        self.lr = lr
        self.n_bins = n_bins
        self.num_labels = num_labels
        self.alpha = alpha
        self.gamma = gamma
        self.useFocalLoss = useFocalLoss
        self.temperature = nn.Parameter(torch.ones(1) * temperature)

    def forward(self, input_ids, attention_mask):
        logits = self.model(input_ids, attention_mask)
        return self.temperature_scale(logits)

    def temperature_scale(self, logits):
        return logits / self.temperature

    def set_temperature(self, validation_loader):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.eval()
        logits_list, labels_list = [], []
        label_prop, loss_fn_name = 'labels', 'BCELogitsLoss'
        if self.num_labels == 1: 
            label_prop = 'blacklist'
            loss_fn_name = f'Focal Loss (gamma: {self.gamma} & alpha: {self.alpha} )'
            
        with torch.no_grad():
            for batch in tqdm(validation_loader, desc="Calibrating temperature"):
                input_ids, attention_mask, labels = batch['input_ids'].to(device), batch['attention_mask'].to(device), batch[label_prop].to(device)
                logits = self.model(input_ids, attention_mask).squeeze()
                logits_list.append(logits)
                labels_list.append(labels)

        logits = torch.cat(logits_list)
        labels = torch.cat(labels_list)

        criterion = nn.BCEWithLogitsLoss()
        if self.useFocalLoss:
            criterion = FocalLoss(alpha=self.alpha, gamma=self.gamma)
        optimizer = torch.optim.LBFGS([self.temperature], lr=self.lr, max_iter=50)

        def eval():
            optimizer.zero_grad()
            loss = criterion(logits / self.temperature, labels.float())
            loss.backward()
            return loss

        optimizer.step(eval)
        
        optimal_temperature = self.temperature.item()
        
        # Calculcate the Class Calibration loss
        class_wise_calibration_error = CECE(num_classes=self.num_labels, n_bins=self.n_bins, norm='l2') # get the count of classes for the experiment and the number of bins (Dataset will be split in 10 parts or nbins)
        
        after_temperature_loss = criterion(self.temperature_scale(logits), labels).item()
        class_wise_calibration_error.update(self.temperature_scale(logits).to(torch.float32), labels.to(torch.int32))
        after_temperature_ece = class_wise_calibration_error.compute()
        print('Optimal temperature: %.3f' % optimal_temperature)
        print(f'After temperature - { loss_fn_name }: { after_temperature_loss }, C/ECE (If num_classes > 1, calculate CECE): { after_temperature_ece}')
        return optimal_temperature