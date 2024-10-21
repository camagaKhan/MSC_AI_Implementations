import torch 
import sys 
from torchmetrics import MetricCollection, Accuracy, Precision, Recall, AUROC, F1Score, ROC, PrecisionRecallCurve, ConfusionMatrix
from torch.utils.data import DataLoader
import tqdm
from torch.utils.data import DataLoader
import pandas as pd
sys.path.append('./')
from LossFunctions.ClassWiseExpectedCalibrationError import CECE
from models.Super_BERT.multi_label.multi_label_implementations.model_skeleton_multilabel_v3 import HateSpeechDataset, HateSpeechTagger, HateSpeechv2Dataset
import pickle
import os

torch.cuda.empty_cache()

MODEL_NAME = 'GroNLP/hateBERT'
BATCH = 16

##### for dataloaders ####
PIN_MEMORY = True
NUM_WORKERS = 0
PREFETCH_FACTOR = None
MAX_LENGTH = 128

# In Malta you can get a hefty fine or get imprisonment if you are charged with Hate Speech. So calibrate that thing!
sys.path.append(os.path.abspath('./models/Super_BERT/multi_label/multi_label_implementations'))

# Load test set only!
test_samples = pd.read_csv('././Data/jigsaw.test.csv') #torch.load('././tokenized/BERT-BASE-CASED/test.pth') # Call the tokenized dataset
test_dataloader = DataLoader(HateSpeechv2Dataset(dataset=test_samples, model_name=MODEL_NAME, max_length=MAX_LENGTH, without_sexual_explict=False), batch_size=BATCH, shuffle=True, num_workers=NUM_WORKERS, prefetch_factor=PREFETCH_FACTOR, pin_memory=PIN_MEMORY) # convert it to a pytorch dataset

folder_name, file_name = 'hateBERT',  'HateBERT_kaiming_128_FL_3_6lbls_jigsaw_2e-05'
checkpoint_path = f'././saved/{folder_name}/{file_name}.model'

# from here: https://stackoverflow.com/questions/67838192/size-mismatch-runtime-error-when-trying-to-load-a-pytorch-model

# Load the best checkpoint of the model based on the validation loss
base_model = torch.load(checkpoint_path)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
base_model.to(device)

THRESHOLD = .5 # .569 <-- dan tajjeb
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
base_model.eval() 
n_bins, NUM_LABELS = 15, 6
predictions, targets = [], []
progress = tqdm.tqdm(test_dataloader, desc='Test batch...', leave=False)
test_log = []
torch.cuda.empty_cache()
with torch.no_grad():
    for _, data in enumerate(progress):            
        input_ids = data['input_ids'].to(device, non_blocking=True)
        attention_mask = data['attention_mask'].to(device, non_blocking=True)
        labels = data['labels'].to(device, non_blocking=True)
        
        pred = base_model(input_ids, attention_mask) 
        
        probs = torch.sigmoid(pred)
        labels_cpu = labels.detach()
        predictions.append(probs)
        targets.append(labels_cpu)

    all_predictions = torch.cat(predictions)
    all_targets = torch.cat(targets)

    results = test_metric(all_predictions.to(torch.float32), all_targets.to(torch.int32))    
    class_wise_calibration_error = CECE(num_classes=NUM_LABELS, n_bins=n_bins, norm='l2') # get the count of classes for the experiment and the number of bins (Dataset will be split in 10 parts or nbins)
    class_wise_calibration_error.update(all_predictions.to(torch.float32), all_targets.to(torch.int32))
    cece_result = class_wise_calibration_error.compute()
    
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
        'CECE' : cece_result.item()
    })
    print(f'\n\nPrinting test metrics.  Accuracy: {results['accuracy'].item()}, F1 (Macro): {results['f1_Macro'].item()}, F1 (Micro): {results['f1_Micro'].item()}, F1 (Weighted) : {results['f1_Weighted'].item()}, AUC: { results['auc_roc_macro'].item() }, precision_macro: {results['precision_macro'].item()}, precision_micro: {results['precision_micro'].item()}, recall_macro: {results['recall_macro'].item()}, recall_micro: {results['recall_micro'].item()}, CECE: {cece_result.item()}')#f'Training Epoch {epoch_id}: Average Training Loss: {average_loss}')
    
    with open(f'././././Metrics_results/HateBert/test/HateBert-ML-jigsaw_6lbls_{'FL'}_{MAX_LENGTH}_testing.pkl', 'wb') as f:
        pickle.dump(test_log, f)







