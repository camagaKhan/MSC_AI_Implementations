import torch 
import sys 
from torchmetrics import MetricCollection, Accuracy, Precision, Recall, AUROC, F1Score, ROC, PrecisionRecallCurve, ConfusionMatrix
from torch.utils.data import DataLoader
import tqdm
from torch.utils.data import DataLoader
import pandas as pd
sys.path.append('./')
from models.Super_BERT.multi_label.multi_label_implementations.model_skeleton_multilabel_v4 import HateSpeechEnsembleDataset
import pickle
import os
import glob 

torch.cuda.empty_cache()

MODEL_NAME_HATEBERT, MODEL_NAME_DISTILBERT = 'GroNLP/hateBERT', 'distilbert-base-cased'
BATCH = 16

##### for dataloaders ####
PIN_MEMORY = True
NUM_WORKERS = 0
PREFETCH_FACTOR = None

# In Malta you can get a hefty fine or get imprisonment if you are charged with Hate Speech. So calibrate that thing!
sys.path.append(os.path.abspath('./models/Super_BERT/multi_label/multi_label_implementations'))

# Load validation set only!
test_samples = pd.read_csv('././Data/jigsaw.test.csv') #torch.load('././tokenized/BERT-BASE-CASED/test.pth') # Call the tokenized dataset
test_dataloader = DataLoader(HateSpeechEnsembleDataset(dataset=test_samples, model_name_1=MODEL_NAME_DISTILBERT, model_name_2=MODEL_NAME_HATEBERT, without_sexual_explict=False), shuffle=True, batch_size=BATCH, num_workers=NUM_WORKERS, prefetch_factor=PREFETCH_FACTOR, pin_memory=PIN_MEMORY) # convert it to a pytorch datasettest_dataloader_distilBERT = DataLoader(HateSpeechv2Dataset(dataset=test_samples, model_name=MODEL_NAME_DISTILBERT, without_sexual_explict=False), batch_size=BATCH, num_workers=NUM_WORKERS, prefetch_factor=PREFETCH_FACTOR, pin_memory=PIN_MEMORY) # convert it to a pytorch dataset

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

HateBERT, DistilBERT = torch.load(f'././saved/hateBERT/HateBERT_kaiming_128_FL_3_6lbls_jigsaw_2e-05.model'), torch.load('././saved/distilbert-base-cased/DistilBERT_jigsaw_FL_3_6lbls_jigsaw.model')

HateBERT.to(device)
DistilBERT.to(device)

HateBERT.eval()
DistilBERT.eval()



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

test_metric.to(device)
#for checkpoint_path in tqdm.tqdm(files, desc="Evaluating Checkpoints for Ensemble Stacking"): 
hs_predictions, hs_targets, predictions, targets, predictions_hatebert, predictions_distilbert = [], [], [], [], [], []
# test_log = []
torch.cuda.empty_cache()
with torch.no_grad():
    progress = tqdm.tqdm(test_dataloader, desc='Test batch...', leave=False)
    for _, data in enumerate(progress):            
        input_ids_distilBERT = data['input_ids_1'].to(device, non_blocking=True)
        attention_mask_distilBERT = data['attention_mask_1'].to(device, non_blocking=True)        
        
        input_ids_HateBERT = data['input_ids_2'].to(device, non_blocking=True)
        attention_mask_HateBERT = data['attention_mask_2'].to(device, non_blocking=True)
        
        labels = data['labels'].to(device, non_blocking=True)
        
        pred_distilBERT = DistilBERT(input_ids_distilBERT, attention_mask_distilBERT)
        probs_distilBERT = torch.sigmoid(pred_distilBERT)
        
        pred_hateBERT = HateBERT(input_ids_HateBERT, attention_mask_HateBERT)
        probs_hateBERT = torch.sigmoid(pred_hateBERT)
        
        
        combined_probs = (probs_hateBERT + probs_distilBERT) / 2

        # Store predictions and targets
        predictions_hatebert.append(probs_hateBERT)
        predictions_distilbert.append(probs_distilBERT)
        hs_predictions.append(combined_probs)
        targets.append(labels.detach())

    stacked_predictions = torch.cat(hs_predictions)
    stacked_targets = torch.cat(targets)
    
# Now calculate the metrics using the stacked predictions and targets
results = test_metric(stacked_predictions.to(torch.float32), stacked_targets.to(torch.int32))
test_log = []
# Log the results
test_log.append({
        #'epoch' : epoch, 
        'accuracy': results['accuracy'].item(),
        #'train_loss' : avg_validation_loss,
        'auc_per_class' : results['auc_per_class'], 
        'auc_roc_macro': results['auc_roc_macro'].item(), 
        'f1_Micro': results['f1_Micro'].item(),
        'f1_Macro': results['f1_Macro'].item(),
        'f1_Weighted': results['f1_Weighted'].item(),
        'f1_per_class': results['f1_per_class'],
        'precision_macro': results['precision_macro'].item(),
        'precision_micro': results['precision_micro'].item(),
        'recall_macro': results['recall_macro'].item(), 
        'recall_micro': results['recall_micro'].item(),
        'precision_per_class_macro': results['precision_per_class_macro'],
        'precision_per_class_micro': results['precision_per_class_micro'],
        'precision_recall_curve': results['precision_recall_curve'],
        'roc_curve': results['roc_curve'],
        'precision_per_class_weighted': results['precision_per_class_weighted'],
        'recall_macro': results['recall_macro'], 
        'recall_micro': results['recall_micro'], 
        'recall_weighted': results['recall_weighted'], 
        'recall_per_class_macro': results['recall_per_class_macro'],
        'recall_per_class_micro': results['recall_per_class_micro'],
        'recall_per_class_weighted': results['recall_per_class_weighted'],
        'precision_recall_curve': results['precision_recall_curve'],
        'confusion_matrix': results['confusion_matrix'],
        #'CECE' : cece_result.item()
    })

# Print the important metrics
print(f'\n\nPrinting test metrics, AUC: {results["auc_roc_macro"].item()}, '
      f'f1 (Macro): {results["f1_Macro"].item()}, f1 (Micro): {results["f1_Micro"].item()}, '
      f'precision_macro: {results["precision_macro"].item()}, precision_micro: {results["precision_micro"].item()}, '
      f'recall_macro: {results["recall_macro"].item()}, recall_micro: {results["recall_micro"].item()}')

with open(f'././././Metrics_results/HateBert/test/hateBert-ML-Cased-jigsaw_6lbls_{'FL'}_ensemble_stacking_training.pkl', 'wb') as f:
    pickle.dump(test_log, f)









