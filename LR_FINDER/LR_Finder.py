import matplotlib.pyplot as plt
import math
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import sys
sys.path.append('./')
from models.super_bert.Model_Skeleton import HateCrimeModel, BinaryHateCrimeDataset, Model_Configuration 
import tqdm

import matplotlib as mpl
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.serif'] = 'Times New Roman'

######### Training Dataset ##########################
torch.cuda.empty_cache() 
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # load the device as cuda if you have a graphic card or cpu if not


MODEL_NAME = 'distilbert-base-cased'
BATCH_SIZE = 16
PIN_MEMORY = True
NUM_WORKERS = 0
PREFETCH_FACTOR = None
LR = 1e-8

training_samples = torch.load('././tokenized/DistilBERT-BASE-CASED/train.pth') # Call the tokenized dataset
training_dataset = BinaryHateCrimeDataset(training_samples) # convert it to a pytorch dataset

train_dataloader = DataLoader(training_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, prefetch_factor=PREFETCH_FACTOR, pin_memory=PIN_MEMORY)

training_samples_length = len(training_samples['input_ids'])
config = Model_Configuration(batch_size= BATCH_SIZE, samples=training_samples_length)

STEP_SIZE = config.calc_step_size()

model = HateCrimeModel(model_name=MODEL_NAME, attn_dropout=.3, hidden_dropout=.3, model_dropout=.3)
binary_optimizer, scheduler, loss_fn = config.load_model(model=model, lr=LR, max_lr=1e-3, weight_decay=1e-4, step_size_up=STEP_SIZE)


model.to(device=device)

def lr_finder(model, dataloader, loss_fn, optimizer, init_lr=LR, final_lr=10, beta=0.98):
    num = len(dataloader) - 1
    mult = (final_lr / init_lr) ** (1/num)
    lr = init_lr
    optimizer.param_groups[0]['lr'] = lr
    avg_loss = 0.
    best_loss = 0.
    batch_num = 0
    losses = []
    log_lrs = []

    progress = tqdm.tqdm(dataloader, desc='Initiating Learning Rate Finder...', leave=False)

    for batch_id, batch in enumerate(progress):
        batch_num += 1
        
        token_list_batch = batch['input_ids'].to(device, non_blocking=True) # use non-blocking to not block the I/O 
        attention_mask_batch = batch['attention_mask'].to(device, non_blocking=True)
        label_batch = batch['blacklist'].to(device, non_blocking=True)
        
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(token_list_batch, attention_mask_batch)
        transformed_prediction_batch = outputs.squeeze()
        loss = loss_fn(transformed_prediction_batch, label_batch)
        
        # Compute the smoothed loss
        avg_loss = beta * avg_loss + (1-beta) * loss.item()
        smoothed_loss = avg_loss / (1 - beta**batch_num)
        
        # Record the best loss
        if batch_num > 1 and smoothed_loss > 4 * best_loss:
            return log_lrs, losses
        
        if smoothed_loss < best_loss or batch_num == 1:
            best_loss = smoothed_loss
        
        # Store the values
        losses.append(smoothed_loss)
        log_lrs.append(math.log10(lr))
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Update the learning rate
        lr *= mult
        optimizer.param_groups[0]['lr'] = lr
        
        if batch_id != 0 and batch_id % 10_000 == 0:
            plt.figure(figsize=(10, 6))
            plt.plot(log_lrs, losses)
            plt.xlabel("Learning Rate (log scale)")
            plt.ylabel("Loss")
            plt.title("Learning Rate Finder")
            plt.savefig(f'./LR_FINDER/lr_progress{batch_id}.png',dpi=300)

    return log_lrs, losses

optimizer = optim.SGD(model.parameters(), lr=1e-8)
log_lrs, losses = lr_finder(model, train_dataloader, loss_fn, optimizer)

plt.figure(figsize=(10, 6))
plt.plot(log_lrs, losses)
plt.xlabel("Learning Rate (log scale)")
plt.ylabel("Loss")
plt.title("Learning Rate Finder")
plt.show()
print('test')