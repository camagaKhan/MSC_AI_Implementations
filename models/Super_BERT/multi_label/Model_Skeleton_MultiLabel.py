import torch 
from torch import nn 
import torch.nn.functional as F 
from transformers import AutoConfig, AutoModelForSequenceClassification
from torch.utils.data import Dataset
from pytorch_lightning import LightningModule, LightningDataModule
from torchmetrics import MetricCollection, Accuracy, Precision, Recall, AUROC, F1Score, ROC, PrecisionRecallCurve
import pickle
import uuid
from torch.utils.data import DataLoader
import sys 
sys.path.append('./')
from LossFunctions.FocalLoss import FocalLoss
from LossFunctions.ClassWiseExpectedCalibrationError import CECE

# myuuid = uuid.uuid4()
# print(myuuid)

class HateCrimeDataset (Dataset):
    def __init__(self, dataset):
        self.data = dataset
        print(self.data['input_ids'].shape)
        self.has_labels = 'labels' in self.data
        
    def __len__(self):
        return self.data['input_ids'].size(0)
    
    def __getitem__(self, index):
        dictionary = {}
        input_ids, attn_mask = self.data['input_ids'][index], self.data['attention_mask'][index]
        if self.has_labels:
            labels = self.data['labels'][index]
            dictionary = dict(index=index, input_ids=input_ids, attention_mask=attn_mask, labels= labels)
        else:
            dictionary = dict(index=index, input_ids=input_ids, attention_mask=attn_mask)
        return dictionary

class HateCrimeDataModule(LightningDataModule):
    def __init__(self, train_dataset, val_dataset, batch_size=16, num_workers=0):
        super(HateCrimeDataModule, self).__init__()
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
    
    def __init__(self, model_name='distilbert-base-cased', step_size_up = 1000, metric_threshold = .5, base_lr = 1e-7, max_lr = 3e-5,  num_labels=7, batch_size=16, attn_dropout = .1 , hidden_dropout = .1, model_dropout = .1, use_focalLoss = False, alpha = .25, gamma = 2., n_bins = 10):
        super(LightHateCrimeModel, self).__init__()
        self.num_labels = num_labels
        self.batch_size = batch_size
        self.base_lr = base_lr
        self.max_lr = max_lr
        self.model_name = model_name
        self.step_size_up = step_size_up
        self.n_bins = n_bins
        # Dropout customization for model =] 
        binary_config = AutoConfig.from_pretrained(self.model_name, num_labels=num_labels)
        binary_config.hidden_dropout_prob = hidden_dropout
        binary_config.attention_probs_dropout_prob = attn_dropout
        self.validation_step_outputs = []
        self.train_step_outputs = []
        
        # Model configuration and updates
        self.classification_model = AutoModelForSequenceClassification.from_pretrained(self.model_name, config = binary_config)
        self.hidden_dim = self.classification_model.config.hidden_size * 4
        
        self.dropout = torch.nn.Dropout(model_dropout)     
        self.model_relu = torch.nn.ReLU()
        self.dense = torch.nn.Linear(self.hidden_dim, self.hidden_dim)
        self.output = torch.nn.Linear(self.hidden_dim, num_labels) # this is where y is outputted 
        
        self.validation_results_all = []
        
        self.loss_fn = nn.BCEWithLogitsLoss()#(weight=self.class_weights)
        if use_focalLoss :
            self.loss_fn = FocalLoss(alpha=alpha, gamma=gamma)
        
        THRESHOLD = metric_threshold
        self.train_metric = MetricCollection({
            'auc_roc_macro': AUROC(num_labels=num_labels, average='macro', task='multilabel'),
            'auc_per_class': AUROC(num_labels=num_labels, average=None, task='multilabel'),
            'f1_Macro': F1Score(task='multilabel', threshold= THRESHOLD, average='macro', num_labels=num_labels),
            'f1_Micro': F1Score(task='multilabel', threshold= THRESHOLD, average='micro', num_labels=num_labels),
            'f1': F1Score(task='multilabel', threshold= THRESHOLD, num_labels=num_labels),
            'f1_per_class': F1Score(num_labels=num_labels, threshold= THRESHOLD, average=None, task='multilabel'),
            'precision_macro': Precision(num_labels=num_labels, threshold= THRESHOLD, average='macro', task='multilabel'),
            'precision_micro': Precision(num_labels=num_labels, threshold= THRESHOLD, average='micro', task='multilabel'),
            'precision_per_class': Precision(num_labels=num_labels, threshold= THRESHOLD, average=None, task='multilabel'),
            'recall_macro': Recall(num_labels=num_labels, threshold= THRESHOLD, average='macro', task='multilabel'), 
            'recall_micro': Recall(num_labels=num_labels, threshold= THRESHOLD,  average='micro', task='multilabel'), 
            'precision_recall_curve': PrecisionRecallCurve(task='multilabel', num_labels=num_labels),
            'roc_curve': ROC(num_labels=num_labels, task='multilabel')
        })
        
        self.validation_metric = self.train_metric.clone()
        
    def forward(self, input_ids, attn_mask):
        outputs = self.classification_model(input_ids, attn_mask, output_hidden_states=True)
        hidden_states = outputs['hidden_states']
                
        features_vector = torch.cat([
            hidden_states[-1][:,0,:], 
            hidden_states[-2][:,0,:],
            hidden_states[-3][:,0,:],
            hidden_states[-4][:,0,:]
            ], dim=-1)
        
        x = self.dropout(features_vector)
        x = self.dense(x) 
        x = self.model_relu(x)
        
        x = self.dropout(x)
        y = self.output(x)
        
        return y
    
    def training_step(self, train_batch, batch_idx):
        self.classification_model.train()
        token_list_batch = train_batch['input_ids']
        attention_mask_batch = train_batch['attention_mask']
        label_batch = train_batch['labels']
        
        # get prediction in squeezed logits 
        outputs = self(token_list_batch, attention_mask_batch)
        logits = outputs.squeeze()
        loss = self.loss_fn(logits.to(torch.float32), label_batch)
        #loss = loss#.item()  
        predictions = torch.sigmoid(logits)
        self.log('train_loss', loss)
        self.train_step_outputs.append({'loss':  loss, 'predictions': predictions, 'targets' : label_batch })
        return loss

    def on_train_epoch_end(self): 
        epoch_id = self.current_epoch
        outputs = self.train_step_outputs
        ml_predictions, ml_targets = torch.cat([x['predictions'] for x in outputs]), torch.cat([x['targets'] for x in outputs])
        results = self.train_metric(ml_predictions.to(torch.float32), ml_targets.to(torch.int32))
        avg_loss = torch.tensor([x['loss'] for x in outputs]).mean()
        
        self.log('auc_roc_macro_t', results['auc_roc_macro'])
        self.log('f1_macro_t', results['f1_Macro'])
        self.log('precision_macro_t', results['precision_macro'])
        self.log('recall_macro_t', results['recall_macro'])

        print('\nPrinting training metrics ...')
        b_metrics = {
           'epoch': epoch_id,
           'loss' : avg_loss,
           'auc_per_class' : results['auc_per_class'], 
           'auc_roc_macro': results['auc_roc_macro'].item(), 
           'f1 Micro': results['f1_Micro'].item(),
           'f1 Macro': results['f1_Macro'].item(),
           'f1_per_class': results['f1_per_class'],
           'precision_macro': results['precision_macro'].item(),
           'precision_micro': results['precision_micro'].item(),
           'recall_macro': results['recall_macro'].item(), 
           'precision_per_class': results['precision_per_class']
        } 
               
        print(b_metrics, '\n')
        
        with open(f'./Metrics_results/{self.model_name}/training/multi-label_{self.model_name}_validation_{epoch_id}_scores_{uuid.uuid4()}.pkl', 'wb') as f:
            pickle.dump(b_metrics, f)
            
        with open(f'./Metrics_results/{self.model_name}/validation/multi-label_{self.model_name}_validation_{epoch_id}_scores_{uuid.uuid4()}.pkl', 'wb') as f:
            pickle.dump(self.validation_results_all, f)   
            
        self.train_step_outputs.clear()
        self.validation_results_all.clear()
        self.train_metric.reset()
        torch.cuda.empty_cache() 
    
    def validation_step(self, validation_batch, batch_idx):
        self.classification_model.eval()
        token_list_batch = validation_batch['input_ids']
        attention_mask_batch = validation_batch['attention_mask']
        label_batch = validation_batch['labels']
        
        outputs = self(token_list_batch, attention_mask_batch)
        logits = outputs.squeeze()
        loss = self.loss_fn(logits.to(torch.float32), label_batch)
        loss = loss.item()  # Reduce loss to a scalar by taking the mean
        predictions = torch.sigmoid(logits)
        self.log('val_loss', loss, prog_bar=True)
        self.validation_step_outputs.append({'loss':  loss, 'predictions': predictions, 'targets' : label_batch })
    
    def on_validation_epoch_end(self): 
        epoch_id = self.current_epoch
        outputs = self.validation_step_outputs
        ml_predictions, ml_targets = torch.cat([x['predictions'] for x in outputs]), torch.cat([x['targets'] for x in outputs])
        results = self.validation_metric(ml_predictions.to(torch.float32), ml_targets.to(torch.int32))
        avg_loss = torch.tensor([x['loss'] for x in outputs]).mean()
        
        class_wise_calibration_error = CECE(num_classes=self.num_labels, n_bins=self.n_bins, norm='l2') # get the count of classes for the experiment and the number of bins (Dataset will be split in 10 parts or nbins)
        class_wise_calibration_error.update(ml_predictions.to(torch.float32), ml_targets.to(torch.int32))
        cece_result = class_wise_calibration_error.compute()
        
        self.log('auc_roc_macro_v', results['auc_roc_macro'])
        self.log('f1_macro_v', results['f1_Macro'])
        self.log('precision_macro_v', results['precision_macro'])
        self.log('recall_macro_v', results['recall_macro'])
        self.log('cece', cece_result)
        
        print('\nPrinting validation metrics ...')
        b_metrics = {
           'epoch' : epoch_id, 
           'loss' : avg_loss,
           'auc_per_class' : results['auc_per_class'], 
           'auc_roc_macro': results['auc_roc_macro'].item(), 
           'f1 Micro': results['f1_Micro'].item(),
           'f1 Macro': results['f1_Macro'].item(),
           'f1_per_class': results['f1_per_class'],
           'precision_macro': results['precision_macro'].item(),
           'precision_micro': results['precision_micro'].item(),
           'recall_macro': results['recall_macro'].item(), 
           'precision_per_class': results['precision_per_class'],
           'C/ECE' : cece_result.item()
        }         
        #self.log(f'validation_{epoch_id}_{uuid.uuid4()}', b_metrics)          
        print(b_metrics, '\n')
        
        self.validation_results_all.append(b_metrics)        
        class_wise_calibration_error.reset()   
        self.validation_metric.reset()
        self.validation_step_outputs.clear()
        torch.cuda.empty_cache()
    
    
    def configure_optimizers(self): 
        optimizer = torch.optim.AdamW(self.classification_model.parameters(), lr=self.base_lr) 
        scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=self.base_lr, max_lr=self.max_lr, step_size_up=self.step_size_up, mode='triangular2', gamma=.9)
        
        scheduler = {
            'scheduler': scheduler,
            'interval': 'step',  # or 'epoch'
            'frequency': 1
        }
        
        return [optimizer], [scheduler]   