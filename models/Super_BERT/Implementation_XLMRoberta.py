import torch 
from pytorch_lightning import Trainer
from Model_SkeletonV2 import BinaryHateCrimeDataset, LightHateCrimeModel, BinaryHateCrimeDataModule
from pytorch_lightning.callbacks import ModelCheckpoint, TQDMProgressBar
from lightning import LightningModule

# print('Is GPU Available?', torch.cuda.is_available())
# print('PyTorch Version:', torch.__version__)

LR = 3e-5 # 2e-5 #1e-5 <-- Learning rate is ok # Initial Learning Rate ; 5e-5 was bad
EPOCHS = 3
num_labels = 7 # removed severe_toxicity
MODEL_NAME = 'xlm-roberta-base'
BATCH_SIZE = 16

######### Training Dataset ##########################
torch.cuda.empty_cache() 
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # load the device as cuda if you have a graphic card or cpu if not

############################################## training ########################################################################

training_samples = torch.load('././tokenized/xlm-roberta-base/train.pth') # Call the tokenized dataset
training_dataset = BinaryHateCrimeDataset(training_samples) # convert it to a pytorch dataset

#train_dataloader = DataLoader(training_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, prefetch_factor=PREFETCH_FACTOR, pin_memory=PIN_MEMORY)

############################################## validation ########################################################################
validation_samples = torch.load('././tokenized/xlm-roberta-base/validation.pth') # Call the tokenized dataset
validation_dataset = BinaryHateCrimeDataset(validation_samples) # convert it to a pytorch dataset

#validation_dataloader = DataLoader(validation_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, prefetch_factor=PREFETCH_FACTOR, pin_memory=PIN_MEMORY)

########################################################### Metrics ############################################################################

training_samples_len = len(training_samples['input_ids'])

def calc_step_size(batch_size, samples):
    if batch_size <= 0:
        raise ValueError("Batch size must be a positive integer")
    return samples // batch_size # integer division

STEP_SIZE_UP = calc_step_size(BATCH_SIZE, training_samples_len)
print('STEPS (FOR CYCLICAL LEARNING RATE)', STEP_SIZE_UP)
print(training_samples['labels'].shape)
blacklist_column = training_samples['labels'][:, -1]
print(blacklist_column.shape)
positives, negatives = torch.sum((blacklist_column == 1)).item(), torch.sum((blacklist_column == 0)).item()
print(len(blacklist_column) == (positives + negatives))


model = LightHateCrimeModel(
    model_name=MODEL_NAME,
    binary_lr=LR, 
    step_size_up=STEP_SIZE_UP,
    batch_size=BATCH_SIZE, 
    attn_dropout=.4,
    hidden_dropout=.3,
    model_dropout=.3, 
    use_focalLoss=True, 
    gamma=3, 
    alpha=.25)


print(isinstance(model, LightningModule))


checkpoint_callback = ModelCheckpoint(
    monitor='val_loss',  # Metric to monitor
    dirpath='./saved/xlm-roberta-base',  # Directory to save the checkpoints
    filename='xlm-roberta-base-{epoch:02d}-{val_loss:.2f}',  # Filename pattern
    save_top_k=3,  # Save the top 3 models
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

