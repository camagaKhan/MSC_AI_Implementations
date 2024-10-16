import torch 
import sys 
from torchmetrics import MetricCollection, Accuracy, Precision, Recall, AUROC, F1Score, ROC, PrecisionRecallCurve, ConfusionMatrix
from torch.utils.data import DataLoader
import tqdm
from torch.utils.data import DataLoader
import pandas as pd
sys.path.append('./')
from LossFunctions.ClassWiseExpectedCalibrationError import CECE
from models.Super_BERT.multi_label.multi_label_implementations.model_skeleton_multilabel_v3 import HateSpeechv2Dataset
import pickle
import os
import glob 

torch.cuda.empty_cache()

MODEL_NAME = 'distilbert-base-uncased'
BATCH = 16

##### for dataloaders ####
PIN_MEMORY = True
NUM_WORKERS = 0
PREFETCH_FACTOR = None
NUM_LABELS, n_bins = 6, 15

# In Malta you can get a hefty fine or get imprisonment if you are charged with Hate Speech. So calibrate that thing!
sys.path.append(os.path.abspath('././models/Super_BERT/multi_label/multi_label_implementations'))

# Load validation set only!
test_samples = pd.read_csv('./././Data/jigsaw.test.csv') #torch.load('././tokenized/BERT-BASE-CASED/test.pth') # Call the tokenized dataset
test_dataloader = DataLoader(HateSpeechv2Dataset(dataset=test_samples, model_name=MODEL_NAME, without_sexual_explict=False), shuffle=True, batch_size=BATCH, num_workers=NUM_WORKERS, prefetch_factor=PREFETCH_FACTOR, pin_memory=PIN_MEMORY) # convert it to a pytorch datasettest_dataloader_distilBERT = DataLoader(HateSpeechv2Dataset(dataset=test_samples, model_name=MODEL_NAME_DISTILBERT, without_sexual_explict=False), batch_size=BATCH, num_workers=NUM_WORKERS, prefetch_factor=PREFETCH_FACTOR, pin_memory=PIN_MEMORY) # convert it to a pytorch dataset

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if os.path.isdir('./././saved/'):
    print("Directory exists")
else:
    print("Directory does not exist")

THRESHOLD = .5
num_labels = 6
test_metric = MetricCollection({
            'accuracy': Accuracy(task="multilabel", threshold=THRESHOLD, num_labels=num_labels),
            'auc_roc_macro': AUROC(num_labels=num_labels, average='macro', task='multilabel'),
            'auc_per_class': AUROC(num_labels=num_labels, average=None, task='multilabel'),
            'f1_Macro': F1Score(task='multilabel', threshold= THRESHOLD, average='macro', num_labels=num_labels),
            'f1_Micro': F1Score(task='multilabel', threshold= THRESHOLD, average='micro', num_labels=num_labels),
            'f1_Weighted': F1Score(task='multilabel', threshold= THRESHOLD, average='weighted', num_labels=num_labels),
            'f1': F1Score(task='multilabel', threshold= THRESHOLD, num_labels=num_labels),
            'f1_per_class': F1Score(num_labels=num_labels, threshold= THRESHOLD, average=None, task='multilabel'),
            'precision_macro': Precision(num_labels=num_labels, threshold= THRESHOLD, average='macro', task='multilabel'),
            'precision_micro': Precision(num_labels=num_labels, threshold= THRESHOLD, average='micro', task='multilabel'),
            'precision_per_class_macro': Precision(num_labels=num_labels, threshold= THRESHOLD, average='macro', task='multilabel'),
            'precision_per_class_micro': Precision(num_labels=num_labels, threshold= THRESHOLD, average='micro', task='multilabel'),
            'precision_per_class_weighted': Precision(num_labels=num_labels, threshold= THRESHOLD, average='weighted', task='multilabel'),
            'recall_macro': Recall(num_labels=num_labels, threshold= THRESHOLD, average='macro', task='multilabel'), 
            'recall_micro': Recall(num_labels=num_labels, threshold= THRESHOLD,  average='micro', task='multilabel'), 
            'recall_weighted': Recall(num_labels=num_labels, threshold= THRESHOLD,  average='weighted', task='multilabel'), 
            'recall_per_class_macro': Recall(num_labels=num_labels, threshold= THRESHOLD, average='macro', task='multilabel'),
            'recall_per_class_micro': Recall(num_labels=num_labels, threshold= THRESHOLD, average='micro', task='multilabel'),
            'recall_per_class_weighted': Recall(num_labels=num_labels, threshold= THRESHOLD, average='weighted', task='multilabel'),
            'precision_recall_curve': PrecisionRecallCurve(task='multilabel', num_labels=num_labels),
            'roc_curve': ROC(num_labels=num_labels, task='multilabel'),
            'confusion_matrix': ConfusionMatrix(threshold=THRESHOLD, num_labels=num_labels, task='multilabel')
        })
all_ensemble_probs, all_labels = [], []

test_metric.to(device)
models = []
for i in range(1, 6):
    checkpoint = f'BERT_FL_16_2_fold_{i}.model'
    model = torch.load(f'./././saved/distilbert-base-uncased/fold/{checkpoint}')
    model.to(device)
    model.eval()
    models.append(model)

torch.cuda.empty_cache() 

class_wise_calibration_error = CECE(num_classes=NUM_LABELS, n_bins=n_bins, norm='l2')
progress = tqdm.tqdm(test_dataloader, desc='Test batch...', leave=False)
for _, data in enumerate(progress) :
    input_ids = data['input_ids'].to(device, non_blocking=True)
    attention_mask = data['attention_mask'].to(device, non_blocking=True)
    labels = data['labels'].to(device, non_blocking=True)
    
    fold_probs = []
    
    for model in models: 
        with torch.no_grad():
            pred = model(input_ids, attention_mask) 
            probs = torch.sigmoid(pred)        
            # Append the probabilities for this fold
            fold_probs.append(probs)
    
    # Stack and average the predictions across all folds
    ensemble_probs = torch.mean(torch.stack(fold_probs), dim=0)  # Average the probabilities
    #ensemble_probs = ensemble_probs.to(device)
    
    # class_wise_calibration_error.update(ensemble_probs.to(torch.float32), labels.to(torch.int32))
    # Store the ensemble probabilities and labels for this batch
    all_ensemble_probs.append(ensemble_probs.cpu())  # Move to CPU to free up GPU memory
    all_labels.append(labels.cpu())  # Move labels to CPU
    # Evaluate using the metric (make sure ensemble_probs and labels are on the same device)
    test_metric.update(ensemble_probs.to(torch.float32), labels.detach().to(torch.int32))

# Concatenate all probabilities and labels for the entire dataset
all_ensemble_probs = torch.cat(all_ensemble_probs, dim=0)  # Concatenate along the batch dimension
all_labels = torch.cat(all_labels, dim=0)   


class_wise_calibration_error.update(all_ensemble_probs.to(torch.float32), all_labels.to(torch.int32))
cece_result = class_wise_calibration_error.compute()
    
# Compute the final results across all batches
final_results = test_metric.compute()

myResults = dict(results = final_results, cece = cece_result)

with open(f'././././Metrics_results/ensemble/distilbert_uncased-ML-Cased-jigsaw_6lbls_{'FL'}_ensemble_averaging_training.pkl', 'wb') as f:
    pickle.dump(myResults, f)

# Print the final evaluation results
print(f"Final Test Metrics: {myResults}")

# Reset the metric for future evaluation if necessary
test_metric.reset()










