from Model_Skeleton import Model_Configuration, HateCrimeModel, BinaryHateCrimeDataset
import torch 
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import GradScaler, autocast
from torchmetrics import MetricCollection, Metric, Accuracy, Precision, Recall, AUROC, F1Score, ROC, PrecisionRecallCurve
import tqdm
import pickle

print('Is GPU Available?', torch.cuda.is_available())
print('PyTorch Version:', torch.__version__)

LR = 2e-5 # 2e-5 #1e-5 <-- Learning rate is ok # Initial Learning Rate ; 5e-5 was bad
EPOCHS = 3
num_labels = 7 # removed severe_toxicity
MODEL_NAME = 'distilbert-base-cased'
PIN_MEMORY = True
NUM_WORKERS = 0
PREFETCH_FACTOR = None
BATCH_SIZE = 16
THRESHOLD = 0.5 # was 0.5

######### Training Dataset ##########################
torch.cuda.empty_cache() 
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # load the device as cuda if you have a graphic card or cpu if not

############################################## training ########################################################################

training_samples = torch.load('././tokenized/DistilBERT-BASE-CASED/train.pth') # Call the tokenized dataset
training_dataset = BinaryHateCrimeDataset(training_samples) # convert it to a pytorch dataset

train_dataloader = DataLoader(training_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, prefetch_factor=PREFETCH_FACTOR, pin_memory=PIN_MEMORY)

############################################## validation ########################################################################
validation_samples = torch.load('././tokenized/BERT-BASE-CASED/validation.pth') # Call the tokenized dataset
validation_dataset = BinaryHateCrimeDataset(validation_samples) # convert it to a pytorch dataset

validation_dataloader = DataLoader(validation_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, prefetch_factor=PREFETCH_FACTOR, pin_memory=PIN_MEMORY)

########################################################### Metrics ############################################################################

binary_metrics = {
    'auc_roc_macro': AUROC(num_labels=1, average='macro', task='binary'),
    'auc_per_class': AUROC(num_labels=1, average=None, task='binary'),
    'f1_Macro': F1Score(task='binary', threshold= THRESHOLD, average='macro', num_labels=1),
    'f1_Micro': F1Score(task='binary', threshold= THRESHOLD, average='micro', num_labels=1),
    'f1': F1Score(task='binary', threshold= THRESHOLD, num_labels=1),
    'f1_per_class': F1Score(num_labels=1, threshold= THRESHOLD, average=None, task='binary'),
    'precision_macro': Precision(num_labels=1, threshold= THRESHOLD, average='macro', task='binary'),
    'precision_micro': Precision(num_labels=1, threshold= THRESHOLD, average='micro', task='binary'),
    'precision_per_class': Precision(num_labels=1, threshold= THRESHOLD, average=None, task='binary'),
    'recall_macro': Recall(num_labels=1, threshold= THRESHOLD, average='macro', task='binary'), 
    'recall_micro': Recall(num_labels=1, threshold= THRESHOLD,  average='micro', task='binary'), 
    'precision_recall_curve': PrecisionRecallCurve(task='binary', num_labels=1),
    'roc_curve': ROC(num_labels=1, task='binary')
}


train_metric = MetricCollection(binary_metrics)
train_metric.to(device)

validation_metric = train_metric.clone()
validation_metric.to(device)

training_samples_length = len(training_samples['input_ids'])
config = Model_Configuration(batch_size= BATCH_SIZE, samples=training_samples_length)

# Returns the step size for cyclical learning rate
STEP_SIZE = 1 #config.calc_step_size()

model = HateCrimeModel(model_name=MODEL_NAME, attn_dropout=.3, hidden_dropout=.3, model_dropout=.3)
binary_optimizer, scheduler, loss_fn = config.load_model(model=model, lr=LR, max_lr=2e-4, weight_decay=1e-4, step_size_up=STEP_SIZE, useStep=True)

model.to(device=device)
scaler= GradScaler()


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
        label_batch = batch['blacklist'].to(device, non_blocking=True)
        
        with autocast(): # https://wandb.ai/wandb_fc/tips/reports/How-To-Use-GradScaler-in-PyTorch--VmlldzoyMTY5MDA5
            #Predict
            y = model(token_list_batch, attention_mask_batch)
            
            y_transformed = y.squeeze()
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

    for batch_id, batch in enumerate(progress):
        
        token_list_batch = batch['input_ids'].to(device, non_blocking=True) # use non-blocking to not block the I/O 
        attention_mask_batch = batch['attention_mask'].to(device, non_blocking=True)
        label_batch = batch['blacklist'].to(device, non_blocking=True)
        
        binary_optimizer.zero_grad() 
        with autocast(): # https://wandb.ai/wandb_fc/tips/reports/How-To-Use-GradScaler-in-PyTorch--VmlldzoyMTY5MDA5
            prediction_batch = model(token_list_batch, attention_mask_batch)
            transformed_prediction_batch = prediction_batch.squeeze()
            
            loss = loss_fn(transformed_prediction_batch, label_batch)
            total_loss += loss.item()
            
        scaler.scale(loss).backward()
        scaler.unscale_(binary_optimizer)  # Unscale the gradients of optimizer's assigned params in-place
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
               
        # Scaler step. Updates the optimizer's params.
        scaler.step(binary_optimizer)        
        # Updates the scale for next iteration.
        scaler.update()
        num_batches += 1       
        
        proba_prediction_batch = torch.sigmoid(transformed_prediction_batch).detach()
        labels_cpu = label_batch.detach()
        predictions.append(proba_prediction_batch)
        targets.append(labels_cpu)
        
        if batch_id != 0 and batch_id % evaluate == 0:
            val_loss_mean = validate_model(epoch_id=epoch_id)
    
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
    progress = tqdm.tqdm(range(1, EPOCHS + 1), desc=f"Training {MODEL_NAME} for {EPOCHS} epochs.", leave=True)
    for epoch in progress:
        stop_training, BEST_VAL_LOSS, NO_IMPROVEMENT = train_model(epoch_id=epoch, evaluate=10_000, BEST_VAL_LOSS=BEST_VAL_LOSS, NO_IMPROVEMENT=NO_IMPROVEMENT)
        if stop_training : 
            break
        val_loss_mean = validate_model(epoch_id=epoch)        
        print ('BEST VAL LOSS',BEST_VAL_LOSS, 'NO IMPROVEMENT', NO_IMPROVEMENT)
        scheduler.step()
        get_lr = scheduler.get_last_lr()
        print('Learning rates:', get_lr)
        learning_rates.append(get_lr)
        
        torch.save(model.state_dict(), f'./saved/{MODEL_NAME}/checkPoints/model_epoch_{epoch}.pt')
        
    with open('./Metrics_results/distilbert-base-cased/training.pkl', 'wb') as f:
        pickle.dump(train_loss_results, f)
        
    with open('./Metrics_results/distilbert-base-cased/learning_rates.pkl', 'wb') as f:
        pickle.dump(learning_rates, f)

    with open('./Metrics_results/distilbert-base-cased/validation.pkl', 'wb') as f:
        pickle.dump(val_loss, f)
        
except RuntimeError as e :
    print(e)  