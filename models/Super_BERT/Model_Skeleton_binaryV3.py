import torch 
from torch import nn 
import torch.nn.functional as F 
from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer
from torch.utils.data import Dataset
from pytorch_lightning import LightningModule, LightningDataModule
from torchmetrics import MetricCollection, Accuracy, Precision, Recall, AUROC, F1Score, ROC, PrecisionRecallCurve, ConfusionMatrix
from torchmetrics.classification import BinaryCalibrationError
import pickle
import uuid
from torch.utils.data import DataLoader
import os
import sys 
sys.path.append('./')
from LossFunctions.FocalLoss import FocalLoss
from LossFunctions.ClassWiseExpectedCalibrationError import CECE

# myuuid = uuid.uuid4()
# print(myuuid)
class BinaryHateCrimeDataset (Dataset):
    def __init__(self, dataset, model_name, max_length=128, has_labels=False, label_name = 'not_offensive'):
        self.data = dataset
        self.model_name = model_name
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.has_labels = has_labels
        self.label_name = label_name
        
    def __len__(self):
        get_length = len(self.data['comment_text'])
        return get_length
    
    def __getitem__(self, index):
        dictionary, item = {}, self.data.iloc[index]
        #input_ids, attn_mask = self.data['input_ids'][index], self.data['attention_mask'][index]
        tokens = self.tokenizer(item['comment_text'], add_special_tokens=True, max_length=self.max_length, padding='max_length', truncation = True, return_tensors='pt')
        if self.has_labels:
            labels = torch.tensor(item[self.label_name].astype(float), dtype=torch.float)
            dictionary = dict(index=index, input_ids=tokens['input_ids'], attention_mask=tokens['attention_mask'], blacklist=labels) # <-- Black List or White List
        else:
            dictionary = dict(index=index, input_ids=tokens['input_ids'], attention_mask=tokens['attention_mask'])
        return dictionary

class BinaryHateCrimeDataModule(LightningDataModule):
    def __init__(self, train_dataset, val_dataset, batch_size=16, num_workers=0):
        super(BinaryHateCrimeDataModule, self).__init__()
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = True
    
    def prepare_data(self):
        # Download data if needed
        pass
    
    def setup(self, stage=None):
        # Split data into train and validation sets
        # Here we assume datasets are already split and passed to the DataModule
        pass
    
    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=True, shuffle=True)
    
    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=True, shuffle=False)    

class LightHateCrimeModel (LightningModule) :
    
    def __init__(self, metrics_folder_name, model_name='distilbert-base-cased', step_size_up = 1000, metric_threshold = .5, binary_base_lr = 1e-7,  binary_lr=2e-5, batch_size=16, attn_dropout = .1 , hidden_dropout = .1, model_dropout = .1, use_focalLoss = False, alpha = .25, gamma = 2., n_bins = 10):
        super(LightHateCrimeModel, self).__init__()
        self.saved_folder_name = metrics_folder_name
        self.binary_labels = 1
        self.batch_size = batch_size
        self.binary_learning_rate = binary_lr
        self.binary_base_lr = binary_base_lr
        self.model_name = model_name
        self.step_size_up = step_size_up
        self.n_bins = n_bins
        # Dropout customization for model =] 
        binary_config = AutoConfig.from_pretrained(self.model_name, num_labels=self.binary_labels)
        binary_config.hidden_dropout_prob = hidden_dropout
        binary_config.attention_probs_dropout_prob = attn_dropout
        self.validation_step_outputs = []
        self.train_step_outputs = []
        
        self.validation_results_all = []
        
        # Model configuration and updates
        self.binary_classification_model = AutoModelForSequenceClassification.from_pretrained(self.model_name, config = binary_config)
        self.hidden_size = self.binary_classification_model.config.hidden_size
        
        self.HateCrimeBlackListDropout = torch.nn.Dropout(model_dropout)
        
        # activation
        #self.model_relu = torch.nn.ReLU()
        #self.classifier = torch.nn.Linear(self.hidden_size, self.binary_labels)
        ## initialize weights for the classifier layer
        #torch.nn.init.kaiming_uniform(self.classifier.weight, nonlinearity='relu')
        
        self.loss_fn_binary = nn.BCEWithLogitsLoss()#(weight=self.class_weights)
        
        if use_focalLoss :
            self.loss_fn_binary = FocalLoss(alpha=alpha, gamma=gamma)
        
        THRESHOLD = metric_threshold
        self.binary_train_metric = MetricCollection({
            'auc_roc_macro': AUROC(average='macro', task='binary'),
            'f1_Macro': F1Score(task='binary', threshold= THRESHOLD, average='macro'),
            'f1': F1Score(task='binary', threshold= THRESHOLD),
            'f1_weighted': F1Score(task='binary', average='weighted', threshold= THRESHOLD),
            'precision_macro': Precision(threshold= THRESHOLD, average='macro', task='binary'),
            'recall_macro': Recall(threshold= THRESHOLD, average='macro', task='binary'), 
            'precision_recall_curve': PrecisionRecallCurve(task='binary'),
            'roc_curve': ROC(task='binary'),
            'accuracy' : Accuracy(task='binary', threshold= THRESHOLD),
            'conf_matrix': ConfusionMatrix(task='binary', threshold=THRESHOLD)
        })
        
        self.binary_validation_metric = self.binary_train_metric.clone()
        
    def forward(self, input_ids, attn_mask):
        outputs =self.binary_classification_model(input_ids, attn_mask)
        outputs = outputs.logits 
        outputs = self.HateCrimeBlackListDropout(outputs) # <--- Before            
        return outputs
    
    def training_step(self, train_batch, batch_idx):
        self.binary_classification_model.train()
        token_list_batch = train_batch['input_ids'].squeeze()
        attention_mask_batch = train_batch['attention_mask'].squeeze()
        label_batch = train_batch['blacklist']
        
        # get prediction in squeezed logits 
        outputs = self(token_list_batch, attention_mask_batch)
        binary_logits = outputs.squeeze()
        loss = self.loss_fn_binary(binary_logits.to(torch.float32), label_batch)
        #loss = loss#.item()  
        binary_predictions = torch.sigmoid(binary_logits)
        self.log('train_loss', loss)
        self.train_step_outputs.append({'loss':  loss, 'binary_predictions': binary_predictions, 'binary_targets' : label_batch })
        return loss

    def on_train_epoch_end(self): 
        epoch_id = self.current_epoch
        outputs = self.train_step_outputs
        b_predictions, b_targets = torch.cat([x['binary_predictions'] for x in outputs]), torch.cat([x['binary_targets'] for x in outputs])
        binary_results = self.binary_train_metric(b_predictions.to(torch.float32), b_targets.to(torch.int32))
        avg_loss = torch.tensor([x['loss'] for x in outputs]).mean()
        
        self.log('auc_roc_macro_t', binary_results['auc_roc_macro'])
        self.log('f1_macro_t', binary_results['f1_Macro'])
        self.log('precision_macro_t', binary_results['precision_macro'])
        self.log('recall_macro_t', binary_results['recall_macro'])

        print('\nPrinting training metrics ...') 
        b_metrics = {
            'epoch' : epoch_id,
            'loss': avg_loss.item(),
            'auc_roc_macro': binary_results['auc_roc_macro'].item(),
            'f1 Macro': binary_results['f1_Macro'].item(),
            'precision_macro': binary_results['precision_macro'].item(),
            'recall_macro': binary_results['recall_macro'].item(),
            'accuracy' : binary_results['accuracy'].item(),
            'conf_matrix': binary_results['conf_matrix'],
            'f1_weighted': binary_results['f1_weighted'].item()
        } 
        
        #self.log(f'training_{epoch_id}', b_metrics)               
        print(b_metrics, '\n')
        
        with open(f'./Metrics_results/{self.saved_folder_name}/training/{self.saved_folder_name}_training_{epoch_id}_scores_{uuid.uuid4()}.pkl', 'wb') as f:
            pickle.dump(b_metrics, f)              
            
        with open(f'./Metrics_results/{self.saved_folder_name}/validation/{self.saved_folder_name}_validation_{epoch_id}_scores_{uuid.uuid4()}.pkl', 'wb') as f:
            pickle.dump(self.validation_results_all, f)          
        
        self.train_step_outputs.clear()
        self.validation_results_all.clear()
        self.binary_train_metric.reset()
        torch.cuda.empty_cache()
    
    def validation_step(self, validation_batch, batch_idx):
        self.binary_classification_model.eval()
        token_list_batch = validation_batch['input_ids'].squeeze()
        attention_mask_batch = validation_batch['attention_mask'].squeeze()
        label_batch = validation_batch['blacklist']
        
        outputs = self(token_list_batch, attention_mask_batch)
        binary_logits = outputs.squeeze()
        loss = self.loss_fn_binary(binary_logits.to(torch.float32), label_batch)
        loss = loss.item()  # Reduce loss to a scalar by taking the mean
        binary_predictions = torch.sigmoid(binary_logits)
        self.log('val_loss', loss, prog_bar=True)
        self.validation_step_outputs.append({'loss':  loss, 'binary_predictions': binary_predictions, 'binary_targets' : label_batch })
    
    def on_validation_epoch_end(self): 
        epoch_id = self.current_epoch
        outputs = self.validation_step_outputs
        b_predictions, b_targets = torch.cat([x['binary_predictions'] for x in outputs]), torch.cat([x['binary_targets'] for x in outputs])
        binary_results = self.binary_validation_metric(b_predictions.to(torch.float32), b_targets.to(torch.int32))
        avg_loss = torch.tensor([x['loss'] for x in outputs]).mean()
        
        # Calculcate the Class Calibration loss
        class_wise_calibration_error = CECE(num_classes=self.binary_labels, n_bins=self.n_bins, norm='l2') # get the count of classes for the experiment and the number of bins (Dataset will be split in 10 parts or nbins)
        class_wise_calibration_error.update(b_predictions.to(torch.float32), b_targets.to(torch.int32))
        cece_result = class_wise_calibration_error.compute()
        
        self.log('auc_roc_macro_v', binary_results['auc_roc_macro'])
        self.log('f1_macro_v', binary_results['f1_Macro'])
        self.log('precision_macro_v', binary_results['precision_macro'])
        self.log('recall_macro_v', binary_results['recall_macro'])
        self.log('cece', cece_result)

        print('\nPrinting validation metrics ...')
        b_metrics = {
            'epoch' : epoch_id,
            'loss' : avg_loss.item(),
            'auc_roc_macro': binary_results['auc_roc_macro'].item(),
            'f1 Macro': binary_results['f1_Macro'].item(),
            'precision_macro': binary_results['precision_macro'].item(),
            'recall_macro': binary_results['recall_macro'].item(),
            'C/ECE' : cece_result.item(),
            'accuracy' : binary_results['accuracy'].item(),
            'conf_matrix': binary_results['conf_matrix'],
            'f1_weighted': binary_results['f1_weighted'].item()
        } 
        #self.log(f'validation_{epoch_id}_{uuid.uuid4()}', b_metrics)          
        print(b_metrics, '\n')
        
        self.validation_results_all.append(b_metrics)        
        class_wise_calibration_error.reset()   
        self.binary_validation_metric.reset()
        self.validation_step_outputs.clear()
        torch.cuda.empty_cache()
    
    
    def configure_optimizers(self): 
        binary_optimizer = torch.optim.AdamW(self.binary_classification_model.parameters(), lr=self.binary_learning_rate) 
        binary_scheduler = torch.optim.lr_scheduler.CyclicLR(binary_optimizer, base_lr=self.binary_base_lr, max_lr=self.binary_learning_rate, step_size_up=self.step_size_up, mode='triangular2', gamma=.9)
        
        scheduler = {
            'scheduler': binary_scheduler,
            'interval': 'step',  # or 'epoch'
            'frequency': 1
        }
        
        return [binary_optimizer], [scheduler]   