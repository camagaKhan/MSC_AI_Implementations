import torch 
from pytorch_lightning import Trainer
import sys
from Model_Skeleton_MultiLabel import HateCrimeDataModule, HateCrimeDataset, LightHateCrimeModel
from pytorch_lightning.callbacks import ModelCheckpoint, TQDMProgressBar
from lightning import LightningModule

# print('Is GPU Available?', torch.cuda.is_available())
# print('PyTorch Version:', torch.__version__)

LR = 2e-5 # 2e-5 #1e-5 <-- Learning rate is ok # Initial Learning Rate ; 5e-5 was bad
EPOCHS = 3
num_labels = 2 # removed severe_toxicity
MODEL_NAME = 'distilbert-base-cased'
BATCH_SIZE = 16

######### Training Dataset ##########################
torch.cuda.empty_cache() 
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # load the device as cuda if you have a graphic card or cpu if not

############################################## training ########################################################################

training_samples = torch.load('././tokenized/DistilBERT-BASE-CASED/multi-label/train_majority.pth') # Call the tokenized dataset
training_dataset = HateCrimeDataset(training_samples) # convert it to a pytorch dataset

#train_dataloader = DataLoader(training_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, prefetch_factor=PREFETCH_FACTOR, pin_memory=PIN_MEMORY)

############################################## validation ########################################################################
validation_samples = torch.load('././tokenized/DistilBERT-BASE-CASED/multi-label/validation_majority.pth') # Call the tokenized dataset
validation_dataset = HateCrimeDataset(validation_samples) # convert it to a pytorch dataset

#validation_dataloader = DataLoader(validation_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, prefetch_factor=PREFETCH_FACTOR, pin_memory=PIN_MEMORY)

########################################################### Metrics ############################################################################

training_samples_len = len(training_samples['input_ids'])

def calc_step_size(batch_size, samples):
    if batch_size <= 0:
        raise ValueError("Batch size must be a positive integer")
    return samples // batch_size # integer division

STEP_SIZE_UP = calc_step_size(BATCH_SIZE, training_samples_len)
print('STEPS (FOR CYCLICAL LEARNING RATE)', STEP_SIZE_UP)


model = LightHateCrimeModel(
    max_lr = LR, 
    num_labels = num_labels,
    step_size_up=STEP_SIZE_UP,
    batch_size=BATCH_SIZE, 
    attn_dropout=.3,
    hidden_dropout=.3,
    use_focalLoss=True, 
    alpha= .25,
    gamma= 3,
    model_dropout=.3)


print(isinstance(model, LightningModule))


checkpoint_callback = ModelCheckpoint(
    monitor='val_loss',  # Metric to monitor
    dirpath='././saved/distilbert-base-cased',  # Directory to save the checkpoints
    filename='multi-label-distilbert-FL-majority-cased-{epoch:02d}-{val_loss:.6f}',  # Filename pattern
    save_top_k=3,  # Save the top 3 models
    mode='min'  # Save the models with the minimum val_loss
)

progress_bar = TQDMProgressBar(refresh_rate=10)  # Customize refresh rate as needed

trainer = Trainer(
    val_check_interval=2_000, # run validation loop every 5000 steps
    max_epochs=EPOCHS, 
    precision=16,  # Enable mixed precision with precision=16
    callbacks=[checkpoint_callback, progress_bar]  # Add the checkpoint callback
)

data_module = HateCrimeDataModule(training_dataset, validation_dataset, batch_size=BATCH_SIZE)

trainer.fit(
        model=model, 
        datamodule=data_module
    )

