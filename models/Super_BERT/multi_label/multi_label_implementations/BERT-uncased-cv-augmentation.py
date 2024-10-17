import torch
from torch.utils.data import DataLoader
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
#from sklearn.model_selection import KFold
import tqdm
import numpy as np
import pandas as pd
import pickle
import os
from torchmetrics import MetricCollection, ConfusionMatrix, Accuracy, Precision, Recall, AUROC, F1Score, ROC, PrecisionRecallCurve
import sys
sys.path.append('./')
from model_skeleton_multilabel_v3 import HateSpeechTagger
from model_skeleton_multilabel_v4 import HateSpeechv2Dataset
from LossFunctions.ClassWiseExpectedCalibrationError import CECE
from LossFunctions.FocalLoss import FocalLoss

# https://github.com/trent-b/iterative-stratification

MODEL_NAME = 'bert-base-uncased'
EPOCHS = 2
LEARNING_RATE = 3e-5
NUM_LABELS = 6
BATCH = 16

##### for dataloaders ####
PIN_MEMORY = True
NUM_WORKERS = 0
PREFETCH_FACTOR = None

THRESHOLD, num_labels = .5, 6
train_metric = MetricCollection({
            'accuracy': Accuracy(task="multilabel", threshold=THRESHOLD, num_labels=num_labels),
            'auc_roc_macro': AUROC(num_labels=num_labels, average='macro', task='multilabel'),
            'auc_per_class': AUROC(num_labels=num_labels, average=None, task='multilabel'),
            'f1_Macro': F1Score(task='multilabel', threshold= THRESHOLD, average='macro', num_labels=num_labels),
            'f1_Micro': F1Score(task='multilabel', threshold= THRESHOLD, average='micro', num_labels=num_labels),
            'f1_Weighted': F1Score(task='multilabel', threshold= THRESHOLD, average='weighted', num_labels=num_labels),
            'f1': F1Score(task='multilabel', threshold= THRESHOLD, num_labels=num_labels),
            'f1_per_class': F1Score(num_labels=num_labels, threshold= THRESHOLD, average=None, task='multilabel'),
            'precision_macro': Precision(num_labels=num_labels, threshold= THRESHOLD, average='macro', task='multilabel'),
            'precision_micro': Precision(num_labels=num_labels, threshold= THRESHOLD, average='micro', task='multilabel'),
            'precision_per_class_macro': Precision(num_labels=num_labels, threshold= THRESHOLD, average='macro', task='multilabel'),
            'precision_per_class_micro': Precision(num_labels=num_labels, threshold= THRESHOLD, average='micro', task='multilabel'),
            'precision_per_class_weighted': Precision(num_labels=num_labels, threshold= THRESHOLD, average='weighted', task='multilabel'),
            'recall_macro': Recall(num_labels=num_labels, threshold= THRESHOLD, average='macro', task='multilabel'), 
            'recall_micro': Recall(num_labels=num_labels, threshold= THRESHOLD,  average='micro', task='multilabel'), 
            'recall_weighted': Recall(num_labels=num_labels, threshold= THRESHOLD,  average='weighted', task='multilabel'), 
            'recall_per_class_macro': Recall(num_labels=num_labels, threshold= THRESHOLD, average='macro', task='multilabel'),
            'recall_per_class_micro': Recall(num_labels=num_labels, threshold= THRESHOLD, average='micro', task='multilabel'),
            'recall_per_class_weighted': Recall(num_labels=num_labels, threshold= THRESHOLD, average='weighted', task='multilabel'),
            'precision_recall_curve': PrecisionRecallCurve(task='multilabel', num_labels=num_labels),
            'roc_curve': ROC(num_labels=num_labels, task='multilabel'),
            'confusion_matrix': ConfusionMatrix(threshold=THRESHOLD, num_labels=num_labels, task='multilabel')
        })

validation_metric = train_metric.clone()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # load the device as cuda if you have a graphic card or cpu if not

train_metric.to(device)
validation_metric.to(device)

# loading embeddings and labels for train and validation
augmentation = pd.read_csv('./././Data/augmented_dataset.csv')
training_pd = pd.read_csv('./././Data/jigsaw.15.train.multi-label.csv')
print('Combined datasets will be in this length: ', len(training_pd) + len(augmentation))
training_pd = pd.concat([training_pd, augmentation], axis=0)
training_pd = training_pd.sample(frac=1, random_state=42).reset_index(drop=True)

labels = ['toxicity', 'obscene', 'sexual_explicit', 'identity_attack', 'insult', 'threat']
y_labels = training_pd[labels].values
dataset = HateSpeechv2Dataset(dataset=training_pd, model_name=MODEL_NAME, without_sexual_explict=False, max_length=128)  

training_log = []
def train_epoch(transformer, fold, criterion, optimizer, train_dl, device, epoch, train_metric = None) :
    avg_training_loss = 0.
    total_loss, num_batches = 0., 0
    all_loss = []
    transformer.train()
    predictions, targets = [], []
    progress = tqdm.tqdm(train_dl, desc=f'Training Epoch {epoch}', leave=False)
    for i, data in enumerate(progress):
        input_ids = data['input_ids'].to(device, non_blocking=True)
        attention_mask =  data['attention_mask'].to(device, non_blocking=True)
        labels = data['labels'].to(device, non_blocking=True)
        
        optimizer.zero_grad()
        
        pred = transformer(input_ids, attention_mask)
        transformed_predictions = pred.squeeze()
        loss = criterion(pred, labels)
        loss.backward()
        all_loss.append(loss)
        total_loss += loss.item()        
        optimizer.step()
        
               
        probs = torch.sigmoid(transformed_predictions).detach()
        labels_cpu = labels.detach()
        predictions.append(probs)
        targets.append(labels_cpu)
        
        num_batches += 1
        torch.nn.utils.clip_grad_norm_(transformer.parameters(), 1.0)
        
    # Update the progress bar
    progress.set_postfix({'batch_loss': loss.item()})
    
    all_predictions = torch.cat(predictions)
    all_targets = torch.cat(targets)
    
    results = train_metric(all_predictions.to(torch.float32), all_targets.to(torch.int32))   
            
    avg_training_loss = total_loss/num_batches
    training_log.append({
        'kFold': fold,
        'epoch' : epoch, 
        'accuracy': results['accuracy'].item(),
        'train_loss' : avg_training_loss,
        'auc_per_class' : results['auc_per_class'], 
        'auc_roc_macro': results['auc_roc_macro'].item(), 
        'f1_Weighted': results['f1_Weighted'].item(),
        'f1_Micro': results['f1_Micro'].item(),
        'f1_Macro': results['f1_Macro'].item(),
        'f1_per_class': results['f1_per_class'],
        'precision_macro': results['precision_macro'].item(),
        'precision_micro': results['precision_micro'].item(),
        'recall_macro': results['recall_macro'].item(), 
        'recall_micro': results['recall_micro'].item(),
        'precision_per_class_macro': results['precision_per_class_macro'],
        'precision_per_class_micro': results['precision_per_class_micro'],
        'precision_recall_curve': results['precision_recall_curve'],
        'roc_curve': results['roc_curve'],
        'precision_per_class_weighted': results['precision_per_class_weighted'],
        'recall_macro': results['recall_macro'], 
        'recall_micro': results['recall_micro'], 
        'recall_weighted': results['recall_weighted'], 
        'recall_per_class_macro': results['recall_per_class_macro'],
        'recall_per_class_micro': results['recall_per_class_micro'],
        'recall_per_class_weighted': results['recall_per_class_weighted'],
        'precision_recall_curve': results['precision_recall_curve'],
        'confusion_matrix': results['confusion_matrix']
    })
    print(f'\n\nPrinting training metrics. Epoch: {epoch}, loss: {avg_training_loss}, Accuracy: {results['accuracy'].item()}, F1 (Macro): {results['f1_Micro'].item()}, F1 (Weighted) : {results['f1_Weighted'].item()},  AUC: { results['auc_roc_macro'].item() }, precision_macro: {results['precision_macro'].item()}, precision_micro: {results['precision_micro'].item()}, recall_macro: {results['recall_macro'].item()}, recall_micro: {results['recall_micro'].item()}')#f'Training Epoch {epoch_id}: Average Training Loss: {average_loss}')
    return avg_training_loss, results

n_bins = 15
validation_log = []
def validate_epoch(transformer, fold, criterion, validation_dl, device, epoch, validation_metric = None):
    predictions, targets = [], []
    avg_validation_loss = 0.
    total_loss, num_batches = 0., 0
    all_loss = []
    transformer.eval()
    progress = tqdm.tqdm(validation_dl, desc='Validation batch...', leave=False)
    with torch.no_grad():
        for _, data in enumerate(progress):
            input_ids = data['input_ids'].to(device, non_blocking=True)
            attention_mask =  data['attention_mask'].to(device, non_blocking=True)
            labels = data['labels'].to(device, non_blocking=True)
            
            pred = transformer(input_ids, attention_mask)
            transformed_predictions = pred.squeeze()
            loss = criterion(transformed_predictions, labels)
            all_loss.append(loss)
            total_loss += loss.item()     
            
            probs = torch.sigmoid(transformed_predictions)
            
            num_batches += 1
            
            labels_cpu = labels.detach()
            
            predictions.append(probs)
            targets.append(labels_cpu)
        
        all_predictions = torch.cat(predictions)
        all_targets = torch.cat(targets)
        
        results = validation_metric(all_predictions.to(torch.float32), all_targets.to(torch.int32))
        
        avg_validation_loss = total_loss/num_batches
        
        class_wise_calibration_error = CECE(num_classes=NUM_LABELS, n_bins=n_bins, norm='l2') # get the count of classes for the experiment and the number of bins (Dataset will be split in 10 parts or nbins)
        class_wise_calibration_error.update(all_predictions.to(torch.float32), all_targets.to(torch.int32))
        cece_result = class_wise_calibration_error.compute()
        
        validation_log.append({
        'kFold': fold,
        'epoch' : epoch, 
        'accuracy': results['accuracy'].item(),
        'train_loss' : avg_validation_loss,
        'auc_per_class' : results['auc_per_class'], 
        'auc_roc_macro': results['auc_roc_macro'].item(), 
        'f1_Micro': results['f1_Micro'].item(),
        'f1_Macro': results['f1_Macro'].item(),
        'f1_Weighted': results['f1_Weighted'].item(),
        'f1_per_class': results['f1_per_class'],
        'precision_macro': results['precision_macro'].item(),
        'precision_micro': results['precision_micro'].item(),
        'recall_macro': results['recall_macro'].item(), 
        'recall_micro': results['recall_micro'].item(),
        'precision_per_class_macro': results['precision_per_class_macro'],
        'precision_per_class_micro': results['precision_per_class_micro'],
        'precision_recall_curve': results['precision_recall_curve'],
        'roc_curve': results['roc_curve'],
        'precision_per_class_weighted': results['precision_per_class_weighted'],
        'recall_macro': results['recall_macro'], 
        'recall_micro': results['recall_micro'], 
        'recall_weighted': results['recall_weighted'], 
        'recall_per_class_macro': results['recall_per_class_macro'],
        'recall_per_class_micro': results['recall_per_class_micro'],
        'recall_per_class_weighted': results['recall_per_class_weighted'],
        'precision_recall_curve': results['precision_recall_curve'],
        'confusion_matrix': results['confusion_matrix'],
        'CECE' : cece_result.item()
    })
        
        
    print(f'\n\nPrinting validation metrics. Epoch: {epoch}, loss: {avg_validation_loss}, Accuracy: {results['accuracy'].item()}, F1 (Macro): {results['f1_Micro'].item()}, F1 (Weighted) : {results['f1_Weighted'].item()}, AUC: { results['auc_roc_macro'].item() }, precision_macro: {results['precision_macro'].item()}, precision_micro: {results['precision_micro'].item()}, recall_macro: {results['recall_macro'].item()}, recall_micro: {results['recall_micro'].item()}, CECE : {cece_result.item()}')#f'Training Epoch {epoch_id}: Average Training Loss: {average_loss}')
    return avg_validation_loss, results


# K-Fold Cross Validation
mlskf = MultilabelStratifiedKFold(n_splits=5, shuffle=True, random_state=42)
fold = 0

checkpoint_dir = "./saved/bert-base-uncased"
os.makedirs(checkpoint_dir, exist_ok=True)



def calc_step_size(batch_size, samples):
    if batch_size <= 0:
        raise ValueError("Batch size must be a positive integer")
    return samples // batch_size # integer division

try:
    torch.cuda.empty_cache()
    
    for fold, (train_index, val_index) in enumerate(
        mlskf.split(np.arange(len(dataset)), y_labels)
        ):
        train_subsampler = torch.utils.data.SubsetRandomSampler(train_index)
        val_subsampler = torch.utils.data.SubsetRandomSampler(val_index)
        train_dataloader, validation_dataloader = DataLoader(dataset, sampler=train_subsampler, drop_last=True, batch_size=BATCH, num_workers=NUM_WORKERS, prefetch_factor=PREFETCH_FACTOR, pin_memory=PIN_MEMORY), DataLoader(dataset, sampler=val_subsampler, drop_last=True, batch_size=BATCH, num_workers=NUM_WORKERS, prefetch_factor=PREFETCH_FACTOR, pin_memory=PIN_MEMORY)
        training_samples_len = len(train_dataloader)
        
        print('training steps: ', training_samples_len)
        print('kfold: ',len(train_dataloader), len(validation_dataloader))
        
        # Let's leave everything as is for the moment
        # We loaded the BERT Model
        # default : alpha .25 and gamma 2
        # before: alpha .35 and gamma 3
        # current: alpha .15 and gamma = 2
        transformer = HateSpeechTagger(model_name=MODEL_NAME, n_classes=NUM_LABELS)
        transformer.to(device=device) # run on cuda
        # set the optimizer
        optimizer = torch.optim.AdamW(params=transformer.parameters(), lr=LEARNING_RATE, weight_decay=0.05) # used to be .05
        criterion = FocalLoss(alpha=.25, gamma=3) #torch.nn.BCELoss() # <-- unlike bce logits loss, this loss function's result will not be influenced by an internal sigmoid function. I might revise FocalLoss
        #scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=1e-7, step_size_up=training_samples_len, max_lr=LEARNING_RATE, mode='triangular2')
        progress =  tqdm.tqdm(range(1,EPOCHS+1), desc='Training epoch...', leave=True)
        
        for epoch in progress:
            # Train
            train_average_loss, train_metrics_computed = train_epoch(transformer=transformer, fold=fold+1, criterion=criterion, optimizer=optimizer, device=device, train_metric=train_metric, train_dl=train_dataloader, epoch=epoch)
            # Validation
            val_average_loss, val_metrics_computed = validate_epoch(transformer=transformer, fold=fold+1, criterion=criterion, validation_dl=validation_dataloader, device=device, validation_metric=validation_metric, epoch=epoch)    
             # save checkpoint
            torch.save(transformer, f'./saved/bert-base-uncased/fold/BERT_augmented_FL_{BATCH}_{epoch}_fold_{fold+1}.model')    
            
            #scheduler.step() # <-- normally we do this
            
            progress.set_description(f"Epoch {epoch}, Mean Validation Loss: {val_average_loss:.4f}")
            torch.cuda.empty_cache()
    
    with open(f'././././Metrics_results/BERT-Base-Uncased/training/BERT-uncased-kfold-augmented-FocalLoss_{BATCH}_training_{LEARNING_RATE}.pkl', 'wb') as f:
        pickle.dump(training_log, f)

    with open(f'././././Metrics_results/BERT-Base-Uncased/validation/BERT-uncased-kfold-augmented-FocalLoss_{BATCH}_validation_{LEARNING_RATE}.pkl', 'wb') as f:
        pickle.dump(validation_log, f)
        
        
    
except RuntimeError as e:
    print(e)
