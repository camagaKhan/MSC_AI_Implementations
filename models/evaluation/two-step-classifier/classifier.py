import pandas as pd 
import torch 
from torch.utils.data import DataLoader, Subset
import sys
import pickle 
import tqdm 
from torchmetrics import MetricCollection, Accuracy, Precision, Recall, AUROC, F1Score, ROC, PrecisionRecallCurve, ConfusionMatrix
import os
sys.path.append('./')
from LossFunctions.ClassWiseExpectedCalibrationError import CECE
from models.Super_BERT.Model_Skeleton_binaryV3 import LightHateCrimeModel, BinaryHateCrimeDataset
from model_skeleton_multilabel_v3 import HateSpeechv2Dataset 

print ('imports ok...')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BINARY_CLASSIFIER_MODEL_NAME = 'distilbert-base-cased' # for the acceptable comment classifier. This classifier blacklists comments which are hs but in truth analyses comments for non-offensive content
MULTI_LABEL_CLASSIFIER = 'bert-base-cased'
BATCH_SIZE = 16
MAX_LENGTH = 128

# Load the jigsaw civil comments 
test_samples = pd.read_csv('./././Data/jigsaw.test.csv')
test_set = BinaryHateCrimeDataset(test_samples, model_name=BINARY_CLASSIFIER_MODEL_NAME, max_length=128, has_labels=True)
test_dataloader = DataLoader(test_set, batch_size=BATCH_SIZE, num_workers=0, pin_memory=True, shuffle=False)

# set the multi-label part of the dataset
test_ml_dataloader = DataLoader(HateSpeechv2Dataset(dataset=test_samples, model_name=MULTI_LABEL_CLASSIFIER, without_sexual_explict=False), shuffle=False, batch_size=BATCH_SIZE, num_workers=0, prefetch_factor=None, pin_memory=True) # convert it to a pytorch datasettest_dataloader_distilBERT = DataLoader(HateSpeechv2Dataset(dataset=test_samples, model_name=MODEL_NAME_DISTILBERT, without_sexual_explict=False), batch_size=BATCH, num_workers=NUM_WORKERS, prefetch_factor=PREFETCH_FACTOR, pin_memory=PIN_MEMORY) # convert it to a pytorch dataset


binary_model, binary_checkpoint = LightHateCrimeModel(model_name=BINARY_CLASSIFIER_MODEL_NAME, metrics_folder_name='two-step-classifier'), torch.load('./././saved/distilbert-base-cased/not-offensive-distilbert-FL-epoch=01-val_loss=0.011651.ckpt')
state_dict = binary_checkpoint['state_dict']

new_state_dict= { k: v for k, v in state_dict.items() if k in binary_model.state_dict() and binary_model.state_dict()[k].size() == v.size() }
binary_model.load_state_dict(new_state_dict, strict=False)
binary_model.to(device)

if os.path.isdir(f'././././Metrics_results/ensemble/'):
    print("Directory exists")
else:
    print("Directory does not exist")

models = []
for i in range(1, 6):
    checkpoint = f'BERT_FL_16_2_fold_{i}.model'
    model = torch.load(f'./././saved/bert-base-cased/fold/{checkpoint}')
    model.to(device)
    model.eval()
    models.append(model)

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

binary_predictions, binary_targets = [], []
progress = tqdm.tqdm(test_dataloader, desc='Test batch...', leave=False)


binary_predictions, binary_targets = [], []
all_ensemble_probs, all_labels = [], []
class_wise_calibration_error = CECE(num_classes=num_labels, n_bins=10, norm='l2')
test_loss = []
for _, batch in enumerate(progress):
        
    token_list_batch = batch['input_ids'].squeeze().to(device, non_blocking=True)
    attention_mask_batch = batch['attention_mask'].squeeze().to(device, non_blocking=True)
    label_batch = batch['blacklist'].squeeze().to(device, non_blocking=True)
    #comments = batch['comments']
    
    # Predict
    binary_outputs = binary_model(token_list_batch, attention_mask_batch)        
    binary_logits = binary_outputs.squeeze()
    binary_pred_probs = torch.sigmoid(binary_logits).detach()
    
    binary_predictions.append(binary_pred_probs)
    binary_targets.append(label_batch.detach())

    # Filter out non-offensive comments and pass offensive ones to the multi-label classifier
    offensive_mask = (binary_pred_probs < THRESHOLD) # remember this classifier detects comments which are not offensive. So if the probability is less than 0.5 then that comment is offensive, hence the < THRESHOLD
    offensive_indices = offensive_mask.nonzero(as_tuple=True)[0]
    
    # offensive_indices_list = offensive_indices.cpu().tolist()
    # print(len(offensive_indices), len(binary_logits), [comments[i] for i in offensive_indices_list])
    #offensive_indices = offensive_indices.to(device, non_blocking=True)
    #ml_progress = tqdm.tqdm(test_ml_dataloader, desc='Analysing comment for additional HS labels...')
    
    filtered_dataset = torch.utils.data.Subset(HateSpeechv2Dataset(dataset=test_samples, model_name=MULTI_LABEL_CLASSIFIER, without_sexual_explict=False), offensive_indices.cpu().tolist())
    # set the multi-label part of the dataset
    test_ml_dataloader = DataLoader(filtered_dataset, shuffle=False, batch_size=BATCH_SIZE, num_workers=0, prefetch_factor=None, pin_memory=True) # convert it to a pytorch datasettest_dataloader_distilBERT = DataLoader(HateSpeechv2Dataset(dataset=test_samples, model_name=MODEL_NAME_DISTILBERT, without_sexual_explict=False), batch_size=BATCH, num_workers=NUM_WORKERS, prefetch_factor=PREFETCH_FACTOR, pin_memory=PIN_MEMORY) # convert it to a pytorch dataset
    #ml_progress = tqdm.tqdm(test_ml_dataloader, desc='Multi-label Classifier Testing...', leave=False)

    if len(offensive_indices) > 0 :
        for mlData in test_ml_dataloader:
            input_ids = mlData['input_ids'].to(device, non_blocking=True)#[offensive_indices]
            attention_mask = mlData['attention_mask'].to(device, non_blocking=True)#[offensive_indices]
            labels = mlData['labels'].to(device, non_blocking=True)#[offensive_indices]
            fold_probs = []
            
            for model in models: 
                with torch.no_grad():
                    pred = model(input_ids, attention_mask) 
                    probs = torch.sigmoid(pred)        
                    # Append the probabilities for this fold
                    fold_probs.append(probs)
            
            # Stack and average the predictions across all folds
            ensemble_probs = torch.mean(torch.stack(fold_probs), dim=0)  # Average the probabilities
            
            # Store the ensemble probabilities and labels for this batch
            all_ensemble_probs.append(ensemble_probs.cpu())  # Move to CPU to free up GPU memory
            all_labels.append(labels.cpu())  # Move labels to CPU
            # Evaluate using the metric (make sure ensemble_probs and labels are on the same device)
            test_metric.update(ensemble_probs.to(torch.float32), labels.detach().to(torch.int32))

try:
    final_results = test_metric.compute()        
    # Concatenate all probabilities and labels for the entire dataset
    all_ensemble_probs = torch.cat(all_ensemble_probs, dim=0)  # Concatenate along the batch dimension
    all_labels = torch.cat(all_labels, dim=0)
    class_wise_calibration_error.update(all_ensemble_probs.to(torch.float32), all_labels.to(torch.int32))
    cece_result = class_wise_calibration_error.compute()
     
    # Compute the final results across all batches
    myResults = dict(results = final_results, cece = cece_result)
    print('results', myResults)


    with open(f'././././Metrics_results/ensemble/two-step-ML-Cased-jigsaw_6lbls_{'FL'}_ensemble_averaging_training.pkl', 'wb') as f:
        pickle.dump(myResults, f)
        
except : 
    final_results = test_metric.compute()  
    myResults = final_results
    with open(f'././././Metrics_results/ensemble/two-step-ML-Cased-jigsaw_6lbls_{'FL'}_ensemble_averaging_training.pkl', 'wb') as f:
        pickle.dump(myResults, f)
            
