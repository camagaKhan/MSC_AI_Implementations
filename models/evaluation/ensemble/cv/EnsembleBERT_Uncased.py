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

MODEL_NAME = 'bert-base-uncased'
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


test_metric.to(device)
models = []
for i in range(1, 6):
    checkpoint = f'BERT_FL_16_2_fold_{i}.model'
    model = torch.load(f'./././saved/bert-base-uncased/fold/{checkpoint}')
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
    # Evaluate using the metric (make sure ensemble_probs and labels are on the same device)
    test_metric.update(ensemble_probs.to(torch.float32), labels.detach().to(torch.int32))
    class_wise_calibration_error.update(ensemble_probs.to(torch.float32), labels.to(torch.int32))
    
# Compute the final results across all batches
final_results = test_metric.compute()
cece_result = class_wise_calibration_error.compute()

myResults = dict(results = final_results, cece = cece_result)

with open(f'././././Metrics_results/ensemble/bert_uncased-ML-Cased-jigsaw_6lbls_{'FL'}_ensemble_averaging_training.pkl', 'wb') as f:
    pickle.dump(myResults, f)

# Print the final evaluation results
print(f"Final Test Metrics: {myResults}")

# Reset the metric for future evaluation if necessary
test_metric.reset()
        

# all_probs, all_true_labels, test_log = [], [], []
# for i in tqdm.tqdm(range(1,6), desc='Processing Folds...'): # 6
#     checkpoint = f'BERT_FL_16_2_fold_{i}.model'
#     model = torch.load(f'./././saved/distilbert-base-cased/fold/{checkpoint}')
#     model.to(device)
#     model.eval()
    
#     torch.cuda.empty_cache()
    
#     fold_probs, fold_true_labels = [], []  # Store predictions for the current fold
    
#     with torch.no_grad():
#         progress = tqdm.tqdm(test_dataloader, desc='Test batch...', leave=False)
#         for _, data in enumerate(progress):
#             input_ids = data['input_ids'].to(device, non_blocking=True)
#             attention_mask = data['attention_mask'].to(device, non_blocking=True)
#             labels = data['labels'].to(device, non_blocking=True)
            
#             pred = model(input_ids, attention_mask) 
            
#             probs = torch.sigmoid(pred)
            
#             # Append the batch probabilities to fold_probs list
#             fold_probs.append(probs.cpu())  # Move to CPU to free up GPU memory
#             fold_true_labels.append(labels.cpu())  # Move labels to CPU
            
#      # Concatenate all batch predictions for this fold
#     fold_probs = torch.cat(fold_probs, dim=0)
#     fold_true_labels = torch.cat(fold_true_labels, dim=0)
    
#     # Append fold predictions to the list for ensembling
#     all_probs.append(fold_probs)
#     all_true_labels.append(fold_true_labels)

# # Convert list to tensor for easier averaging or other operations
# ensemble_probs = torch.mean(torch.stack(all_probs), dim=0)  # Averaging across folds
# true_labels = torch.cat(all_true_labels, dim=0)  # Concatenate all true labels

# ensemble_probs = ensemble_probs.to(device)
# true_labels = true_labels.to(device)

# results = test_metric(ensemble_probs.to(torch.float32), true_labels.to(torch.int32))   
    
# class_wise_calibration_error = CECE(num_classes=NUM_LABELS, n_bins=n_bins, norm='l2') # get the count of classes for the experiment and the number of bins (Dataset will be split in 10 parts or nbins)
# class_wise_calibration_error.update(ensemble_probs.to(torch.float32), true_labels.to(torch.int32))
# cece_result = class_wise_calibration_error.compute()

# test_log.append({
#         #'epoch' : epoch, 
#         'accuracy': results['accuracy'].item(),
#         #'train_loss' : avg_validation_loss,
#         'auc_per_class' : results['auc_per_class'], 
#         'auc_roc_macro': results['auc_roc_macro'].item(), 
#         'f1_Micro': results['f1_Micro'].item(),
#         'f1_Macro': results['f1_Macro'].item(),
#         'f1_Weighted': results['f1_Weighted'].item(),
#         'f1_per_class': results['f1_per_class'],
#         'precision_macro': results['precision_macro'].item(),
#         'precision_micro': results['precision_micro'].item(),
#         'recall_macro': results['recall_macro'].item(), 
#         'recall_micro': results['recall_micro'].item(),
#         'precision_per_class_macro': results['precision_per_class_macro'],
#         'precision_per_class_micro': results['precision_per_class_micro'],
#         'precision_recall_curve': results['precision_recall_curve'],
#         'roc_curve': results['roc_curve'],
#         'precision_per_class_weighted': results['precision_per_class_weighted'],
#         'recall_macro': results['recall_macro'], 
#         'recall_micro': results['recall_micro'], 
#         'recall_weighted': results['recall_weighted'], 
#         'recall_per_class_macro': results['recall_per_class_macro'],
#         'recall_per_class_micro': results['recall_per_class_micro'],
#         'recall_per_class_weighted': results['recall_per_class_weighted'],
#         'precision_recall_curve': results['precision_recall_curve'],
#         'confusion_matrix': results['confusion_matrix'],
#         'CECE' : cece_result.item()
#     })

# # Print the important metrics
# print(f'\n\nPrinting test metrics, AUC: {results["auc_roc_macro"].item()}, '
#       f'f1 (Macro): {results["f1_Macro"].item()}, f1 (Micro): {results["f1_Micro"].item()}, '
#       f'precision_macro: {results["precision_macro"].item()}, precision_micro: {results["precision_micro"].item()}, '
#       f'recall_macro: {results["recall_macro"].item()}, recall_micro: {results["recall_micro"].item()}')

# with open(f'././././Metrics_results/ensemble/distilbert_cased-ML-Cased-jigsaw_6lbls_{'FL'}_ensemble_averaging_training.pkl', 'wb') as f:
#     pickle.dump(test_log, f)









