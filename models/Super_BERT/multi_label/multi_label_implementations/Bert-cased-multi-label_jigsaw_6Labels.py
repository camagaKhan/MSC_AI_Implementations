import torch
from torch.utils.data import DataLoader
import tqdm
import sys 
from torchmetrics import MetricCollection, ConfusionMatrix, Metric, Accuracy, Precision, Recall, AUROC, F1Score, ROC, PrecisionRecallCurve
import pickle
sys.path.append('./')
from model_skeleton_multilabel_v3 import HateSpeechDataset, HateSpeechTagger, HateSpeechv2Dataset
from LossFunctions.ClassWiseExpectedCalibrationError import CECE
from LossFunctions.FocalLoss import FocalLoss
from sklearn.utils import class_weight
import numpy as np
import pandas as pd

MODEL_NAME = 'bert-base-cased'
EPOCHS = 3
LEARNING_RATE = 3e-5
NUM_LABELS = 6
BATCH = 16

##### for dataloaders ####
PIN_MEMORY = True
NUM_WORKERS = 0
PREFETCH_FACTOR = None
MAX_LENGTH = 128
loss_fn_name = 'FL'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # load the device as cuda if you have a graphic card or cpu if not

tokenizer_folder_name = 'BERT-BASE-CASED'
# loading embeddings and labels for train and validation
#train, validation = torch.load(f'././././tokenized/{tokenizer_folder_name}/multi-label/train_wiki_280.pth'), torch.load(f'././././tokenized/{tokenizer_folder_name}/multi-label/validation_wiki_280.pth')

training_pd, validation_pd = pd.read_csv('././Data/jigsaw.15.train.multi-label.csv'), pd.read_csv('././Data/jigsaw.validation.multi-label_mixed.csv')

train_dl, validation_dl = DataLoader(HateSpeechv2Dataset(dataset=training_pd, model_name=MODEL_NAME, without_sexual_explict=False, max_length=MAX_LENGTH), batch_size=BATCH, shuffle=True, num_workers=NUM_WORKERS, prefetch_factor=PREFETCH_FACTOR, pin_memory=PIN_MEMORY), DataLoader(HateSpeechv2Dataset(dataset=validation_pd, model_name=MODEL_NAME, without_sexual_explict=False, max_length=MAX_LENGTH), batch_size=BATCH, shuffle=True,num_workers=NUM_WORKERS, prefetch_factor=PREFETCH_FACTOR, pin_memory=PIN_MEMORY)

transformer = HateSpeechTagger(model_name=MODEL_NAME, n_classes=NUM_LABELS, attn_dropout=.4, model_dropout=.4)
transformer.to(device=device) # run on cuda
# set the optimizer
optimizer = torch.optim.AdamW(params=transformer.parameters(), lr=LEARNING_RATE)#, weight_decay=0.0001, eps=1e-8)

# Convert labels to binary (if not already) and flatten the array
binary_labels = training_pd[['toxicity', 'obscene', 'sexual_explicit', 'identity_attack', 'insult', 'threat']].to_numpy()

# Compute class weights for each class
class_weights = []
for i in range(binary_labels.shape[1]):
    y = binary_labels[:, i]
    unique_classes = np.unique(y)
    weight = class_weight.compute_class_weight(class_weight='balanced',
                                        classes=unique_classes,
                                        y=y)
    print(weight)
    if len(weight) == 2:
        class_weights.append(weight[1])  # Append the weight for class '1'
    else:
        class_weights.append(weight[0])  # Append the single weight element # weight[0] corresponds to the weight for class '1'

    
print('Class weights', class_weights)

pos_weights = torch.tensor(class_weights).to(device=device)
criterion = torch.nn.BCEWithLogitsLoss()

if loss_fn_name == 'FL':
    criterion = FocalLoss(alpha=.25, gamma=3)#, pos_weights=pos_weights) 

THRESHOLD, num_labels = .5, NUM_LABELS # 8 was ok
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

train_metric.to(device)
validation_metric.to(device)

total_steps = len(train_dl) * EPOCHS

# Set the number of warmup steps
warmup_steps = int(0.1 * total_steps)
print('warm up steps', warmup_steps)

# total_steps = len(train_dl) * EPOCHS
# scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=1e-5, 
#                                               max_lr=LEARNING_RATE, 
#                                               step_size_up=warmup_steps, 
#                                               mode='triangular2'
#                                               )  #get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

training_log = []
def train_epoch(epoch, check_interval = 10_000) :
    avg_training_loss = 0.
    total_loss, num_batches = 0., 0
    all_loss = []
    transformer.train(True)
    predictions, targets = [], []
    progress = tqdm.tqdm(train_dl, desc=f'Training Epoch {epoch}', leave=False)
    for i, data in enumerate(progress):
        input_ids = data['input_ids'].to(device, non_blocking=True)
        attention_mask =  data['attention_mask'].to(device, non_blocking=True)
        labels = data['labels'].to(device, non_blocking=True)       
        
        optimizer.zero_grad()
        
        pred = transformer(input_ids, attention_mask)
        loss = criterion(pred, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(transformer.parameters(), 1.0)
        optimizer.step()
              
        total_loss += loss.item()
        all_loss.append(loss)
        
        probs = torch.sigmoid(pred) 
        labels_cpu = labels.detach()
        predictions.append(probs)
        targets.append(labels_cpu)
        
        num_batches += 1    
        
        if i != 0 and i % check_interval == 0:
            validate_epoch(epoch=epoch)        
            # Update the progress bar
            progress.set_postfix({'batch_loss': loss.item()})
            
        
        
            
    all_predictions = torch.cat(predictions)
    all_targets = torch.cat(targets)
    
    results = train_metric(all_predictions.to(torch.float32), all_targets.to(torch.int32))   
            
    avg_training_loss = total_loss/num_batches
    training_log.append({
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
    print(f'\n\nPrinting training metrics. Epoch: {epoch}, loss: {avg_training_loss}, AUC: { results['auc_roc_macro'].item() }, precision_macro: {results['precision_macro'].item()}, precision_micro: {results['precision_micro'].item()}, recall_macro: {results['recall_macro'].item()}, recall_micro: {results['recall_micro'].item()}')#f'Training Epoch {epoch_id}: Average Training Loss: {average_loss}')
    #f'Training Epoch {epoch_id}: Average Training Loss: {average_loss}')


validation_log, n_bins, class_wise_calibration_error = [], 15, []
def validate_epoch(epoch):
    avg_validation_loss = 0.
    total_loss, num_batches = 0., 0
    all_loss = []
    transformer.eval()
    predictions, targets = [], []
    progress = tqdm.tqdm(validation_dl, desc='Validation batch...', leave=False)
    with torch.no_grad():
        for _, data in enumerate(progress):
            input_ids = data['input_ids'].to(device, non_blocking=True)
            attention_mask =  data['attention_mask'].to(device, non_blocking=True)
            labels = data['labels'].to(device, non_blocking=True)
            
            pred = transformer(input_ids, attention_mask)
            #transformed_predictions = pred.squeeze()
            loss = criterion(pred, labels)
            all_loss.append(loss)
            total_loss += loss.item()     
            
            probs = torch.sigmoid(pred)
            labels_cpu = labels.detach()
            predictions.append(probs)
            targets.append(labels_cpu)
            
            num_batches += 1
        
        all_predictions = torch.cat(predictions)
        all_targets = torch.cat(targets)
        
        results = validation_metric(all_predictions.to(torch.float32), all_targets.to(torch.int32))    
        avg_validation_loss = total_loss/num_batches
        
        class_wise_calibration_error = CECE(num_classes=NUM_LABELS, n_bins=n_bins, norm='l2') # get the count of classes for the experiment and the number of bins (Dataset will be split in 10 parts or nbins)
        class_wise_calibration_error.update(all_predictions.to(torch.float32), all_targets.to(torch.int32))
        cece_result = class_wise_calibration_error.compute()
        
        validation_log.append({
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
        print(f'\n\nPrinting validation metrics. Epoch: {epoch}, loss: {avg_validation_loss}, AUC: { results['auc_roc_macro'].item() }, precision_macro: {results['precision_macro'].item()}, precision_micro: {results['precision_micro'].item()}, recall_macro: {results['recall_macro'].item()}, recall_micro: {results['recall_micro'].item()}, CECE: {cece_result.item()}')#f'Training Epoch {epoch_id}: Average Training Loss: {average_loss}')
        return avg_validation_loss



try:
    torch.cuda.empty_cache()
    progress = tqdm.tqdm(range(1, EPOCHS + 1), desc='Training Epoch...', leave=True) 
   
    for epoch in progress:
       # Start training
       train_epoch(epoch=epoch, check_interval=10_000)
       
       # validate epoch
       validate_epoch(epoch=epoch)
       
       torch.save(transformer, f'././././saved/bert-base-cased/BERT_{MAX_LENGTH}_{loss_fn_name}_{epoch}_6lbls_jigsaw_{LEARNING_RATE}_MAX_POOLING.model')
       
       torch.cuda.empty_cache() # always empty cache before you start a new epoch
    
    with open(f'././././Metrics_results/BERT-Base-cased/training/BERT-Cased_{MAX_LENGTH}_{loss_fn_name}_jigsaw_6lbls_training_{LEARNING_RATE}_MAX_POOLING.pkl', 'wb') as f:
        pickle.dump(training_log, f)

    with open(f'././././Metrics_results/BERT-Base-cased/validation/BERT-Cased_{MAX_LENGTH}_{loss_fn_name}_jigsaw_6lbls_validation_{LEARNING_RATE}_MAX_POOLING.pkl', 'wb') as f:
        pickle.dump(validation_log, f)
   
except RuntimeError as e:
    print(e)