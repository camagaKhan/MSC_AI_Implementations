import pickle 

file = './Metrics_results/mBERTu/test/mBERTu-FocalLoss_test.pkl'
with open(file, 'rb') as file:
    data = pickle.load(file)

metrics = data[0]
required_metrics = dict(accuracy = metrics['all_metrics']['accuracy'], f1=metrics['all_metrics']['f1_Macro'], macro_precision=metrics['all_metrics']['precision_macro'], macro_recall=metrics['all_metrics']['recall_macro'], 
                        AUC = metrics['all_metrics']['auc_roc_macro'])    
print('\n', required_metrics)

#tensor([[ 21897,   1491],
        #[ 50958, 218782]], device='cuda:0')
        
        
file = './Metrics_results/bert-base-cased/test/BERT-binary-FocalLoss_test.pkl'
with open(file, 'rb') as file:
    data = pickle.load(file)

metrics = data[0]
required_metrics = dict(accuracy = metrics['all_metrics']['accuracy'], f1=metrics['all_metrics']['f1_Macro'], macro_precision=metrics['all_metrics']['precision_macro'], macro_recall=metrics['all_metrics']['recall_macro'], 
                        AUC = metrics['all_metrics']['auc_roc_macro'])    
print('\n', required_metrics)

# tensor([[ 14848,   1040],
#         [ 23614, 154926]], device='cuda:0')

#blacklist-jigsaw-DistilBERT-binary-FocalLoss_test.pkl

file = './Metrics_results/distilbert-base-cased/test/blacklist-jigsaw-DistilBERT-binary-FocalLoss_test.pkl'
with open(file, 'rb') as file:
    data = pickle.load(file)

metrics = data[0]
required_metrics = dict(accuracy = metrics['all_metrics']['accuracy'], f1=metrics['all_metrics']['f1_Macro'], macro_precision=metrics['all_metrics']['precision_macro'], macro_recall=metrics['all_metrics']['recall_macro'], 
                        AUC = metrics['all_metrics']['auc_roc_macro'])    
print('\n', required_metrics)
        
        
#BERT-binary-FocalLoss_test.pkl

# tensor([[175770,   2770],
#         [  6768,   9120]], device='cuda:0')