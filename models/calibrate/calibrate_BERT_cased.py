from temperature_scaling import TemperatureScaling
from torch.utils.data import DataLoader
import torch
import sys
sys.path.append('./')
from models.Super_BERT.Model_SkeletonV2 import LightHateCrimeModel, BinaryHateCrimeDataset

BATCH_SIZE = 16
LR = 3e-5 # 2e-5 #1e-5 <-- Learning rate is ok # Initial Learning Rate ; 5e-5 was bad
EPOCHS = 3
num_labels = 7 # removed severe_toxicity
MODEL_NAME = 'bert-base-cased'
BATCH_SIZE = 16

# In Malta you can get a hefty fine or get imprisonment if you are charged with Hate Speech. So calibrate that thing!

# Load validation set only!
validation_samples = torch.load('././tokenized/BERT-BASE-CASED/validation.pth') # Call the tokenized dataset
validation_dataset = BinaryHateCrimeDataset(validation_samples) # convert it to a pytorch dataset

val_dataloader = DataLoader(validation_dataset, batch_size=BATCH_SIZE, num_workers=0, pin_memory=True, shuffle=False)

folder_name, file_name = 'bert-base-cased',  'bert-cased-epoch=02-val_loss=0.0128'
checkpoint_path = f'./saved/{folder_name}/{file_name}.ckpt'

# from here: https://stackoverflow.com/questions/67838192/size-mismatch-runtime-error-when-trying-to-load-a-pytorch-model

# Load the best checkpoint of the model based on the validation loss
base_model = LightHateCrimeModel(model_name=MODEL_NAME)
checkpoint = torch.load(checkpoint_path)
state_dict = checkpoint['state_dict']

new_state_dict= { k: v for k, v in state_dict.items() if k in base_model.state_dict() and base_model.state_dict()[k].size() == v.size() }
base_model.load_state_dict(new_state_dict, strict=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
base_model.to(device)

calibrated_model = TemperatureScaling(base_model, temperature=1.5, lr=.01, num_labels=1, n_bins=10, alpha=.25, gamma=3, useFocalLoss=True)
calibrated_model.to(device)


temperature = calibrated_model.set_temperature(val_dataloader)


