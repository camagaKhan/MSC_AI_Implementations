import torch 
from pytorch_lightning import Trainer
import pandas as pd
from Model_Skeleton_binaryV3 import BinaryHateCrimeDataset, LightHateCrimeModel, BinaryHateCrimeDataModule
from pytorch_lightning.callbacks import ModelCheckpoint, TQDMProgressBar
from lightning import LightningModule

# print('Is GPU Available?', torch.cuda.is_available())
# print('PyTorch Version:', torch.__version__)

LR = 3e-5 # 2e-5 #1e-5 <-- Learning rate is ok # Initial Learning Rate ; 5e-5 was bad
EPOCHS = 3
num_labels = 7 # removed severe_toxicity
MODEL_NAME = 'distilbert-base-cased'
BATCH_SIZE = 16
MAX_LENGTH = 250

######### Training Dataset ##########################
torch.cuda.empty_cache() 
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # load the device as cuda if you have a graphic card or cpu if not

############################################## training ########################################################################

training_samples = pd.read_csv('././Data/jigsaw.train.csv') #torch.load('././tokenized/DistilBERT-BASE-CASED/train.pth') # Call the tokenized dataset
#training_samples['blacklist'] = 1 
#training_samples.loc[training_samples['not_offensive'] == 1, 'blacklist'] = 0
training_dataset = BinaryHateCrimeDataset(training_samples, model_name=MODEL_NAME, max_length=MAX_LENGTH, has_labels=True, label_name='not_offensive') # convert it to a pytorch dataset

#train_dataloader = DataLoader(training_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, prefetch_factor=PREFETCH_FACTOR, pin_memory=PIN_MEMORY)

############################################## validation ########################################################################
validation_samples = pd.read_csv('././Data/jigsaw.validation.csv') #torch.load('././tokenized/DistilBERT-BASE-CASED/validation.pth') # Call the tokenized dataset
#validation_samples['blacklist'] = 1 
#validation_samples.loc[validation_samples['not_offensive'] == 1, 'blacklist'] = 0
validation_dataset = BinaryHateCrimeDataset(validation_samples, model_name=MODEL_NAME, max_length=MAX_LENGTH, has_labels=True, label_name='not_offensive') # convert it to a pytorch dataset

#validation_dataloader = DataLoader(validation_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, prefetch_factor=PREFETCH_FACTOR, pin_memory=PIN_MEMORY)

########################################################### Metrics ############################################################################

training_samples_len = len(training_samples)

def calc_step_size(batch_size, samples):
    if batch_size <= 0:
        raise ValueError("Batch size must be a positive integer")
    return samples // batch_size # integer division

STEP_SIZE_UP = calc_step_size(BATCH_SIZE, training_samples_len)
print('STEPS (FOR CYCLICAL LEARNING RATE)', STEP_SIZE_UP)

model = LightHateCrimeModel(
    metrics_folder_name='distilbert-base-cased',
    model_name=MODEL_NAME,
    binary_lr=LR, 
    step_size_up=STEP_SIZE_UP,
    batch_size=BATCH_SIZE, 
    attn_dropout=.4,
    hidden_dropout=.3,
    use_focalLoss=True, 
    alpha=.25, 
    gamma=3,
    model_dropout=.2)


print(isinstance(model, LightningModule))


checkpoint_callback = ModelCheckpoint(
    monitor='val_loss',  # Metric to monitor
    dirpath='./saved/distilbert-base-cased',  # Directory to save the checkpoints
    filename='not-offensive-distilbert-FL-{epoch:02d}-{val_loss:.6f}',  # Filename pattern
    save_top_k=4,  # Save the top 4 models
    mode='min'  # Save the models with the minimum val_loss
)

progress_bar = TQDMProgressBar(refresh_rate=10)  # Customize refresh rate as needed

trainer = Trainer(
    val_check_interval=10_000, # run validation loop every 5000 steps
    max_epochs=3, 
    precision=16,  # Enable mixed precision with precision=16
    callbacks=[checkpoint_callback, progress_bar]  # Add the checkpoint callback
)

data_module = BinaryHateCrimeDataModule(training_dataset, validation_dataset, batch_size=BATCH_SIZE)

trainer.fit(
        model=model, 
        datamodule=data_module
    )

