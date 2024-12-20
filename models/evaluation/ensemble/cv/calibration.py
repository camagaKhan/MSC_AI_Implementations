import pandas as pd 
import sys
sys.path.append('./')
from models.calibrate.temperature_scaling import TemperatureScaling 
from model_skeleton_multilabel_v3 import HateSpeechTagger
from model_skeleton_multilabel_v4 import HateSpeechv2Dataset
from torch.utils.data import DataLoader
import torch 
import pickle as pkl
import os

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Imports are okay')

directory_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'temperature_values')

# Check if the directory exists
if os.path.isdir(directory_path):
    print("Directory exists")
else:
    # If the directory does not exist, create it
    print("Directory does not exist, creating it now.")
    os.makedirs(directory_path)

temp_scale = 1.

validation_pd = pd.read_csv('././././Data/jigsaw.validation.multi-label_mixed.csv')
MODEL_NAME, MAX_LENGTH, BATCH = 'distilbert-base-cased', 128, 16
PIN_MEMORY = True
NUM_WORKERS = 0
PREFETCH_FACTOR = None

validation_dl = DataLoader(HateSpeechv2Dataset(dataset=validation_pd, model_name=MODEL_NAME, without_sexual_explict=False, max_length=MAX_LENGTH), batch_size=BATCH, shuffle=True,num_workers=NUM_WORKERS, prefetch_factor=PREFETCH_FACTOR, pin_memory=PIN_MEMORY)


# let's calibrate multi-label distilBERT # this is good
models, model_name, temperatures = [], 'distilbert-base-cased', []
for i in range(1, 6):
    checkpoint = f'BERT_FL_16_2_fold_{i}.model'
    base_model = torch.load(f'./././saved/distilbert-base-cased/fold/{checkpoint}')
    base_model.to(device)
    base_model.eval()    
    calibrated_model = TemperatureScaling(base_model, temperature=temp_scale, lr=0.1, num_labels=6, n_bins=6, alpha=.35, gamma=3, useFocalLoss=True, max_iter=200, norm='l2')
    calibrated_model.to(device)
    temperature = calibrated_model.set_temperature(validation_dl)
    temperatures.append(
        dict(chkpt_name = checkpoint, temperature =temperature)
    )
    
with open (f'{directory_path}\\{model_name}_temp.pkl', 'wb') as file :
    pkl.dump(temperatures, file)

print ('temperatures', temperatures)


temp_scale = 1.6
models, model_name, temperatures = [], 'distilbert-base-uncased', []
for i in range(1, 6):
    checkpoint = f'BERT_FL_16_2_fold_{i}.model'
    base_model = torch.load(f'./././saved/distilbert-base-uncased/fold/{checkpoint}')
    base_model.to(device)
    base_model.eval()    
    calibrated_model = TemperatureScaling(base_model, temperature=temp_scale, lr=0.02, num_labels=6, n_bins=10, alpha=.25, gamma=2, useFocalLoss=True, max_iter=200, norm='l2')
    calibrated_model.to(device)
    temperature = calibrated_model.set_temperature(validation_dl)
    temperatures.append(
        dict(chkpt_name = checkpoint, temperature =temperature)
    )

with open (f'{directory_path}\\{model_name}_temp.pkl', 'wb') as file :
    pkl.dump(temperatures, file)

print ('temperatures', temperatures)

temp_scale = 1.0
models, model_name, temperatures = [], 'bert-base-uncased', []
for i in range(1, 6):
    checkpoint = f'BERT_FL_16_2_fold_{i}.model'
    base_model = torch.load(f'./././saved/bert-base-uncased/fold/{checkpoint}')
    base_model.to(device)
    base_model.eval()    
    calibrated_model = TemperatureScaling(base_model, temperature=temp_scale, lr=0.04, num_labels=6, n_bins=10, alpha=.25, gamma=2, useFocalLoss=True, max_iter=200, norm='l2')
    calibrated_model.to(device)
    temperature = calibrated_model.set_temperature(validation_dl)
    temperatures.append(
        dict(chkpt_name = checkpoint, temperature =temperature)
    )
    
with open (f'{directory_path}\\{model_name}_temp.pkl', 'wb') as file :
    pkl.dump(temperatures, file)

print ('temperatures', temperatures)

temp_scale = 1.0
models, model_name, temperatures = [], 'bert-base-cased', []
for i in range(1, 6):
    checkpoint = f'BERT_FL_16_2_fold_{i}.model'
    base_model = torch.load(f'./././saved/bert-base-cased/fold/{checkpoint}') # This is good
    base_model.to(device)
    base_model.eval()    
    calibrated_model = TemperatureScaling(base_model, temperature=temp_scale, lr=0.04, num_labels=6, n_bins=10, alpha=.25, gamma=2, useFocalLoss=True, max_iter=200, norm='l2')
    calibrated_model.to(device)
    temperature = calibrated_model.set_temperature(validation_dl)
    temperatures.append(
        dict(chkpt_name = checkpoint, temperature =temperature)
    )
    
with open (f'{directory_path}\\{model_name}_temp.pkl', 'wb') as file :
    pkl.dump(temperatures, file)

print ('temperatures', temperatures)


temp_scale = 1.0
models, model_name, temperatures = [], 'roberta-base', []
for i in range(1, 6):
    checkpoint = f'RoBERTa_16_2_fold_{i}.model'
    base_model = torch.load(f'./././saved/roberta-base/fold/{checkpoint}') # This is good
    base_model.to(device)
    base_model.eval()    
    calibrated_model = TemperatureScaling(base_model, temperature=temp_scale, lr=0.1, num_labels=6, n_bins=10, alpha=.25, gamma=2, useFocalLoss=True, max_iter=200, norm='l2')
    calibrated_model.to(device)
    temperature = calibrated_model.set_temperature(validation_dl)
    temperatures.append(
        dict(chkpt_name = checkpoint, temperature =temperature)
    )
    
with open (f'{directory_path}\\{model_name}_temp.pkl', 'wb') as file :
    pkl.dump(temperatures, file)
print ('temperatures', temperatures)


temp_scale = 1.0
models, model_name, temperatures = [], 'hateBERT', []
for i in range(1, 6):
    checkpoint = f'hateBERT_FL_2_fold_{i}.model'
    base_model = torch.load(f'./././saved/hateBERT/fold/{checkpoint}') # This is good
    base_model.to(device)
    base_model.eval()    
    calibrated_model = TemperatureScaling(base_model, temperature=temp_scale, lr=0.02, num_labels=6, n_bins=10, alpha=.25, gamma=2, useFocalLoss=True, max_iter=200, norm='l2')
    calibrated_model.to(device)
    temperature = calibrated_model.set_temperature(validation_dl)
    temperatures.append(
        dict(chkpt_name = checkpoint, temperature =temperature)
    )
    
with open (f'{directory_path}\\{model_name}_temp.pkl', 'wb') as file :
    pkl.dump(temperatures, file)

print ('temperatures', temperatures)