import torch 
import sys 
from torchmetrics import MetricCollection, Accuracy, Precision, Recall, AUROC, F1Score, ROC, PrecisionRecallCurve, ConfusionMatrix
from torch.utils.data import DataLoader
import tqdm
from torch.utils.data import DataLoader
import pandas as pd
sys.path.append('./')
from models.Super_BERT.Model_Skeleton_binaryV3 import LightHateCrimeModel, BinaryHateCrimeDataset
import pickle

BATCH_SIZE = 16
MODEL_NAME = 'MLRS/mBERTu'
MAX_LENGTH = 250

# In Malta you can get a hefty fine or get imprisonment if you are charged with Hate Speech. So calibrate that thing!

# Load validation set only!
test_samples = pd.read_csv('././Data/multi-lingual/hs_test.csv') #torch.load('././tokenized/DistilBERT-BASE-CASED/validation.pth') # Call the tokenized dataset
test_dataset = BinaryHateCrimeDataset(test_samples, model_name=MODEL_NAME, max_length=MAX_LENGTH, has_labels=True) # convert it to a pytorch dataset

test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, num_workers=0, pin_memory=True, shuffle=False)

folder_name, file_name = 'mBERTu',  'mBERTu-FL-epoch=00-val_loss=0.019247'
checkpoint_path = f'././saved/{folder_name}/{file_name}.ckpt'

# from here: https://stackoverflow.com/questions/67838192/size-mismatch-runtime-error-when-trying-to-load-a-pytorch-model

# Load the best checkpoint of the model based on the validation loss
base_model = LightHateCrimeModel(model_name=MODEL_NAME, metrics_folder_name='mBERTu')
checkpoint = torch.load(checkpoint_path)
state_dict = checkpoint['state_dict']

new_state_dict= { k: v for k, v in state_dict.items() if k in base_model.state_dict() and base_model.state_dict()[k].size() == v.size() }
base_model.load_state_dict(new_state_dict, strict=False)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
base_model.to(device)

THRESHOLD = .5
test_metric = MetricCollection({
            'accuracy': Accuracy(threshold=THRESHOLD, task='binary'),
            'auc_roc_macro': AUROC(average='macro', task='binary'),
            'f1_Weighted': F1Score(task='binary', threshold= THRESHOLD, average='weighted'),
            'f1_Macro': F1Score(task='binary', threshold= THRESHOLD, average='macro'),
            'f1_Micro': F1Score(task='binary', threshold= THRESHOLD, average='micro'),
            'f1': F1Score(task='binary', threshold= THRESHOLD),
            'precision_macro': Precision(threshold= THRESHOLD, average='macro', task='binary'),
            'recall_macro': Recall(threshold= THRESHOLD, average='macro', task='binary'), 
            'precision_recall_curve': PrecisionRecallCurve(task='binary'),
            'roc_curve': ROC(task='binary'),
            'confusion_matrix': ConfusionMatrix(threshold=THRESHOLD, task='binary')
        })

test_metric.to(device)


base_model.eval() 


predictions, targets = [], []
progress = tqdm.tqdm(test_dataloader, desc='Test batch...', leave=False)

test_loss = []
for _, batch in enumerate(progress):
        
    token_list_batch = batch['input_ids'].squeeze().to(device, non_blocking=True)
    attention_mask_batch = batch['attention_mask'].squeeze().to(device, non_blocking=True)
    label_batch = batch['blacklist'].to(device, non_blocking=True)
    
    #Predict
    outputs = base_model(token_list_batch, attention_mask_batch)        
    logits = outputs.squeeze()
    #y_transformed = y.squeeze()         
                
    predicted_probabilities = torch.sigmoid(logits).detach()
    labels_cpu = label_batch.detach()
        
    predictions.append(predicted_probabilities)
    targets.append(labels_cpu)
    
all_predictions = torch.cat(predictions)
all_targets = torch.cat(targets)
    
metrics_computed = test_metric(all_predictions.to(torch.float32), all_targets.to(torch.int32))
print({
        'auc_roc_macro': metrics_computed['auc_roc_macro'].item(),
        'f1 Macro': metrics_computed['f1_Macro'].item(),
        'precision_macro': metrics_computed['precision_macro'].item(),
        'recall_macro': metrics_computed['recall_macro'].item()
      }, '\n')
test_loss.append({ 'macro_auc' : metrics_computed['auc_roc_macro'].item(), 'all_metrics' : metrics_computed })   
test_metric.reset()


keys = test_loss[0].keys()
with open('././Metrics_results/mBERTu/test/mBERTu-FocalLoss_test.pkl', 'wb') as f:
    pickle.dump(test_loss, f)




