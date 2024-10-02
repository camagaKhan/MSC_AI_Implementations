import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForSequenceClassification, AutoConfig
from torch.cuda.amp import GradScaler, autocast
from torchmetrics import MetricCollection, Metric, Accuracy, Precision, Recall, AUROC, F1Score, ROC, PrecisionRecallCurve
import tqdm
import pickle
import sys 
sys.path.append('./')
from LossFunctions.FocalLoss import FocalLoss

print('Is GPU Available?', torch.cuda.is_available())
print('PyTorch Version:', torch.__version__)

LR = 2e-5 # 2e-5 #1e-5 <-- Learning rate is ok # Initial Learning Rate ; 5e-5 was bad
EPOCHS = 3
num_labels = 7 # removed severe_toxicity
MODEL_NAME = 'bert-base-cased'
BATCH_SIZE = 32 # 16 # from 32
PIN_MEMORY = True
NUM_WORKERS = 0
PREFETCH_FACTOR = None

class HateSpeechDataset (Dataset) :
    
    def __init__(self, dataset) -> None:
        super().__init__()
        self.data = dataset
        self.has_labels = 'labels' in self.data
        
    def __len__(self):
        return self.data['input_ids'].size(0)
    
    
    def __getitem__(self, index):
        dictionary = {}
        input_ids, attn_mask = self.data['input_ids'][index], self.data['attention_mask'][index]
        if self.has_labels:
            labels = self.data['labels'][index]
            dictionary = dict(index=index, input_ids=input_ids, attention_mask=attn_mask, labels=labels)
        else:
            dictionary = dict(index=index, input_ids=input_ids, attention_mask=attn_mask)
        return dictionary
    
torch.cuda.empty_cache() 
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # load the device as cuda if you have a graphic card or cpu if not

############################################## training ########################################################################

training_samples = torch.load('././tokenized/BERT-BASE-CASED/train.pth') # Call the tokenized dataset
training_dataset = HateSpeechDataset(training_samples) # convert it to a pytorch dataset

train_dataloader = DataLoader(training_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, prefetch_factor=PREFETCH_FACTOR, pin_memory=PIN_MEMORY)

############################################## validation ########################################################################
validation_samples = torch.load('././tokenized/BERT-BASE-CASED/validation.pth') # Call the tokenized dataset
validation_dataset = HateSpeechDataset(validation_samples) # convert it to a pytorch dataset

validation_dataloader = DataLoader(validation_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, prefetch_factor=PREFETCH_FACTOR, pin_memory=PIN_MEMORY)

########################################################### Metrics ############################################################################
THRESHOLD = 0.5 # was 0.5
metrics = {
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
}


train_metric = MetricCollection(metrics)
train_metric.to(device)

validation_metric = train_metric.clone()
validation_metric.to(device)

##################################################### Actual Training Implementation ###################################################
# Great results with alpha .65 and gamma=4
# Used Focal Loss. Get equal precision and recall

def load_model (model_name = 'bert-base-cased', num_labels=7, use_FocalLoss = False, alpha=.65, gamma=4, weight_decay=.01, hidden_dropout=.3, attention_dropout = .35):
    # This will try to minimize overfitting
    config = AutoConfig.from_pretrained(model_name, num_labels = num_labels)
    config.hidden_dropout_prob = hidden_dropout
    config.attention_probs_dropout_prob = attention_dropout
    model = AutoModelForSequenceClassification.from_pretrained(model_name, config = config)
    optimizer, loss_fn = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=weight_decay), torch.nn.BCEWithLogitsLoss().cuda()
    
    # Try this later on... (If you use this, comment line 93 and 94)
    # for layer in model.bert.encoder.layer:
    #     layer.attention.self.dropout = torch.nn.Dropout(p=dropout)
    #     layer.attention.output.dropout = torch.nn.Dropout(p=dropout)
    #     layer.output.dropout = torch.nn.Dropout(p=dropout)
    
    if use_FocalLoss :
        loss_fn = FocalLoss(alpha=alpha, gamma=gamma).cuda()
    return model, optimizer, loss_fn
#                     [        X                   ]
weight_decay_values = [1e-3, 1e-4, 1e-5, 1e-6, 1e-7] # check these weight decay values
model_name = 'bert-base-cased'
model, optimizer, loss_fn = load_model(model_name, use_FocalLoss=True, weight_decay=1e-4)
model.to(device)
scaler = GradScaler()

########################################################### Add Scheduler ############################################################################
training_samples_length = len(training_samples['input_ids'])
print('training samples', training_samples_length)
def calc_step_size(samples, batch_size):
    if batch_size <= 0:
        raise ValueError("Batch size must be a positive integer")
    return samples // batch_size # integer division

step_size_up = calc_step_size(samples=training_samples_length, batch_size=BATCH_SIZE) # <-- step_up_size was 2000 before I created this function

scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=LR, max_lr=2e-4,
                     step_size_up=step_size_up, mode='triangular2', cycle_momentum=False)


########################################################### Implementing early stop ############################################################################


val_loss = []
@torch.no_grad
def validate_model (epoch_id = None) :
    model.eval()     
    predictions, targets = [], []
    total_loss, num_batches = 0, 0
    progress = tqdm.tqdm(validation_dataloader, desc='Validation batch...', leave=False)
    for _, batch in enumerate(progress):
        
        token_list_batch = batch['input_ids'].to(device, non_blocking=True)
        attention_mask_batch = batch['attention_mask'].to(device, non_blocking=True)
        label_batch = batch['labels'].to(device, non_blocking=True)
        
        with autocast(): # https://wandb.ai/wandb_fc/tips/reports/How-To-Use-GradScaler-in-PyTorch--VmlldzoyMTY5MDA5
            #Predict
            y = model(token_list_batch, attention_mask_batch)
            
            y_transformed = y.logits.squeeze()
            #Loss
            loss = loss_fn(
                y_transformed.to(torch.float32),
                label_batch.to(torch.float32)
            )
            total_loss += loss.item()
            num_batches += 1            
                    
        proba_y_batch = torch.sigmoid(y_transformed).detach()
        labels_cpu = label_batch.detach()
        predictions.append(proba_y_batch)
        targets.append(labels_cpu)
        
    all_predictions = torch.cat(predictions)
    all_targets = torch.cat(targets)
    
    metrics_computed = validation_metric(all_predictions.to(torch.float32), all_targets.to(torch.int32))
    print('Printing metrics ...')
    print({'auc_per_class' : metrics_computed['auc_per_class'], 
           'auc_roc_macro': metrics_computed['auc_roc_macro'].item(), 
           'f1 Micro': metrics_computed['f1_Micro'].item(),
           'f1 Macro': metrics_computed['f1_Macro'].item(),
           'f1_per_class': metrics_computed['f1_per_class'],
           'precision_macro': metrics_computed['precision_macro'].item(),
           'precision_micro': metrics_computed['precision_micro'].item(),
           'recall_macro': metrics_computed['recall_macro'].item(), 
           'precision_per_class': metrics_computed['precision_per_class']
           }, '\n')
    loss_mean = total_loss / num_batches
    # Compute AUROC
    print(f' Epoch id ({epoch_id}): Validation Loss in cycle', loss_mean, ', AUROC (Macro)', metrics_computed['auc_roc_macro'].item())
    val_loss.append({ 'epoch' : epoch_id, 'loss' : loss_mean, 'macro_auc' : metrics_computed['auc_roc_macro'].item(), 'all_metrics': metrics_computed })
    validation_metric.reset()
    return loss_mean

train_loss_results = []
def train_model (epoch_id = None, evaluate=10_000, PATIENCE= 3, BEST_VAL_LOSS = float('inf'), NO_IMPROVEMENT = 0): 
    model.train()
    progress = tqdm.tqdm(train_dataloader, desc='Training batch...', leave=False)
    total_loss, num_batches = 0, 0
    predictions, targets = [], []
    stop_training = False
    #BEST_VAL_LOSS, NO_IMPROVEMENT = float('inf')

    for batch_id, batch in enumerate(progress):
        
        token_list_batch = batch['input_ids'].to(device, non_blocking=True) # use non-blocking to not block the I/O 
        attention_mask_batch = batch['attention_mask'].to(device, non_blocking=True)
        label_batch = batch['labels'].to(device, non_blocking=True)
        
        optimizer.zero_grad() 
        
        with autocast(): # https://wandb.ai/wandb_fc/tips/reports/How-To-Use-GradScaler-in-PyTorch--VmlldzoyMTY5MDA5
            prediction_batch = model(token_list_batch, attention_mask_batch)
            transformed_prediction_batch = prediction_batch.logits.squeeze()
            
            loss = loss_fn(transformed_prediction_batch, label_batch)
            total_loss += loss.item()
            
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)  # Unscale the gradients of optimizer's assigned params in-place
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
               
        # Scaler step. Updates the optimizer's params.
        scaler.step(optimizer)        
        # Updates the scale for next iteration.
        scaler.update()
        num_batches += 1       
        
        proba_prediction_batch = torch.sigmoid(transformed_prediction_batch).detach()
        labels_cpu = label_batch.detach()
        predictions.append(proba_prediction_batch)
        targets.append(labels_cpu)
        
        if batch_id != 0 and batch_id % evaluate == 0:
            val_loss_mean = validate_model(epoch_id=epoch_id)
            
            # if val_loss_mean < BEST_VAL_LOSS:
            #     BEST_VAL_LOSS = val_loss_mean
            #     NO_IMPROVEMENT = 0
            #     torch.save(model.state_dict(), f'./saved/{model_name}/checkPoints/Best-BERT-CASED-Checkpoint.pt')
            # else :
            #     NO_IMPROVEMENT += 1
            # print('How many times has validation increased? ', NO_IMPROVEMENT)
            # if NO_IMPROVEMENT >= PATIENCE:
            #     print(f"Early stopping at epoch {epoch}")
            #     stop_training = True
            #     break
            
            #torch.save(model.state_dict(), f'./saved/{model_name}/checkPoints/checkpoint_epoch_{epoch_id}.pt')
    
    all_predictions = torch.cat(predictions)
    all_targets = torch.cat(targets)
    
    metrics_computed = train_metric(all_predictions.to(torch.float32), all_targets.to(torch.int32))    
    print('Printing metrics ...')
    print({'auc_per_class' : metrics_computed['auc_per_class'], 
           'auc_roc_macro': metrics_computed['auc_roc_macro'].item(), 
           'f1 Micro': metrics_computed['f1_Micro'].item(),
           'f1 Macro': metrics_computed['f1_Macro'].item(),
           'f1_per_class': metrics_computed['f1_per_class'],
           'precision_macro': metrics_computed['precision_macro'].item(),
           'precision_micro': metrics_computed['precision_micro'].item(),
           'recall_macro': metrics_computed['recall_macro'].item(), 
           'precision_per_class': metrics_computed['precision_per_class']
           }, '\n')
    average_loss = total_loss / num_batches
    train_loss_results.append({ 'epoch' : epoch_id, 'loss' : average_loss, 'macro_auc' : metrics_computed['auc_roc_macro'].item(), 'all_metrics': metrics_computed })
    print(f'Training Epoch {epoch_id}: Average Training Loss: {average_loss}')
    train_metric.reset()
    return stop_training, BEST_VAL_LOSS, NO_IMPROVEMENT
       
        
        
try:
    BEST_VAL_LOSS, NO_IMPROVEMENT = float('inf'), 0
    learning_rates = []
    torch.cuda.empty_cache()
    progress = tqdm.tqdm(range(1, EPOCHS + 1), desc=f"Training {model_name} for {EPOCHS} epochs.", leave=True)
    for epoch in progress:
        stop_training, BEST_VAL_LOSS, NO_IMPROVEMENT = train_model(epoch_id=epoch, evaluate=4_000, BEST_VAL_LOSS=BEST_VAL_LOSS, NO_IMPROVEMENT=NO_IMPROVEMENT)
        if stop_training : 
            break
        val_loss_mean = validate_model(epoch_id=epoch)        
        print ('BEST VAL LOSS',BEST_VAL_LOSS, 'NO IMPROVEMENT', NO_IMPROVEMENT)
        scheduler.step()
        get_lr = scheduler.get_last_lr()
        print('Learning rates:', get_lr)
        learning_rates.append(get_lr)
        
        torch.save(model.state_dict(), f'./saved/{model_name}/checkPoints/model_epoch_{epoch}.pt')
        
    with open('./Metrics_results/BERT-Base-Cased/BERT_Based_Cased_cycleLR_training.pkl', 'wb') as f:
        pickle.dump(train_loss_results, f)
        
    with open('./Metrics_results/BERT-Base-Cased/BERT_Based_Cased_cycleLR_learning_rates.pkl', 'wb') as f:
        pickle.dump(learning_rates, f)

    with open('./Metrics_results/BERT-Base-Cased/BERT_Based_Cased_cycleLR_validation.pkl', 'wb') as f:
        pickle.dump(val_loss, f)
        
except RuntimeError as e :
    print(e)   

    