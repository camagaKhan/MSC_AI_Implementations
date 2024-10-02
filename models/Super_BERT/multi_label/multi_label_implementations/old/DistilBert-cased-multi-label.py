import torch
from torch.utils.data import DataLoader
import tqdm
import sys 
sys.path.append('./')
from model_skeleton_multilabel_v3 import HateSpeechDataset, HateSpeechTagger
from LossFunctions.FocalLoss import FocalLoss

MODEL_NAME = 'distilbert-base-cased'
EPOCHS = 2
LEARNING_RATE = 2e-5
NUM_LABELS = 6
BATCH = 32

##### for dataloaders ####
PIN_MEMORY = True
NUM_WORKERS = 0
PREFETCH_FACTOR = None

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # load the device as cuda if you have a graphic card or cpu if not

tokenizer_folder_name = 'DistilBERT-BASE-CASED'
# loading embeddings and labels for train and validation
train, validation = torch.load(f'././././tokenized/{tokenizer_folder_name}/multi-label/train_280.pth'), torch.load(f'././././tokenized/{tokenizer_folder_name}/multi-label/validation_280.pth')


train_dl, validation_dl = DataLoader(HateSpeechDataset(dataset=train), batch_size=BATCH, shuffle=True, num_workers=NUM_WORKERS, prefetch_factor=PREFETCH_FACTOR, pin_memory=PIN_MEMORY), DataLoader(HateSpeechDataset(dataset=validation), batch_size=BATCH, shuffle=True,num_workers=NUM_WORKERS, prefetch_factor=PREFETCH_FACTOR, pin_memory=PIN_MEMORY)

transformer = HateSpeechTagger(model_name=MODEL_NAME, n_classes=NUM_LABELS)
transformer.to(device=device) # run on cuda
# set the optimizer
optimizer = torch.optim.AdamW(params=transformer.parameters(), lr=LEARNING_RATE, weight_decay=0.03)

criterion = torch.nn.BCEWithLogitsLoss()  #FocalLoss(alpha=.25, gamma=3) #torch.nn.BCELoss() # <-- unlike bce logits loss, this loss function's result will not be influenced by an internal sigmoid function. I might revise FocalLoss

training_log = []
def train_epoch(epoch, check_interval = 10_000) :
    avg_training_loss = 0.
    total_loss, num_batches = 0., 0
    all_loss = []
    transformer.train()
    progress = tqdm.tqdm(train_dl, desc=f'Training Epoch {epoch}', leave=False)
    for i, data in enumerate(progress):
        input_ids = data['input_ids'].to(device, non_blocking=True)
        attention_mask =  data['attention_mask'].to(device, non_blocking=True)
        labels = data['labels'].to(device, non_blocking=True)
        
        optimizer.zero_grad()
        
        predictions = transformer(input_ids, attention_mask)
        loss = criterion(predictions, labels)
        loss.backward()
        all_loss.append(loss)
        total_loss += loss.item()        
        optimizer.step()
        
        probs = torch.sigmoid(predictions) 
        
        num_batches += 1
        torch.nn.utils.clip_grad_norm_(transformer.parameters(), 1.0)
        if i != 0 and i % check_interval == 0:
            validate_epoch(epoch=epoch)
        
            # Update the progress bar
            progress.set_postfix({'batch_loss': loss.item()})
            
    avg_training_loss = total_loss/num_batches
    training_log.append({
        'epoch' : epoch, 
        'train_loss_averaged' : avg_training_loss
    })
    print(f'\n\nPrinting training metrics. Epoch: {epoch}, loss: {avg_training_loss}')#f'Training Epoch {epoch_id}: Average Training Loss: {average_loss}')


validation_log = []
def validate_epoch(epoch):
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
            
            predictions = transformer(input_ids, attention_mask)
            transformed_predictions = predictions.squeeze()
            loss = criterion(transformed_predictions, labels)
            all_loss.append(loss)
            total_loss += loss.item()     
            
            probs = torch.sigmoid(transformed_predictions)
            
            num_batches += 1
                
        avg_validation_loss = total_loss/num_batches
        validation_log.append({
            'epoch' : epoch,
            #'loss_results' : all_loss, 
            'validation_loss_averaged' : avg_validation_loss
        })
        print(f'\n\nPrinting validation metrics. Epoch: {epoch}, Validation loss: {avg_validation_loss}')#f'Training Epoch {epoch_id}: Average Training Loss: {average_loss}')
        return avg_validation_loss



try:
   torch.cuda.empty_cache()
   progress = tqdm.tqdm(range(1, EPOCHS + 1), desc='Training Epoch...', leave=True) 
   
   for epoch in progress:
       # Start training
       train_epoch(epoch=epoch, check_interval=5_000)
       
       # validate epoch
       validate_epoch(epoch=epoch)
       
       torch.cuda.empty_cache() # always empty cache before you start a new epoch
   
except RuntimeError as e:
    print(e)