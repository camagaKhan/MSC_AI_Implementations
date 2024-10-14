import pickle 

MODEL_NAME, MODE, FILE_NAME = 'bert-base-cased', 'test', 'bert-ML-Cased-jigsaw_6lbls_BCE__training'
with open(f'./Metrics_results/{MODEL_NAME}/{MODE}/{FILE_NAME}.pkl', 'rb') as file:
    data = pickle.load(file)
    
metrics = data[0]
required_metrics = dict(Model=MODEL_NAME, micro_precision = metrics['precision_micro'], 
                        macro_precision=metrics['precision_macro'], micro_recall = metrics['recall_micro'], macro_recall=metrics['recall_macro'], 
                        F1_per_class = metrics['f1_per_class'], auc_per_class=metrics['auc_per_class'])    
print('\n', MODEL_NAME, required_metrics)


MODEL_NAME, MODE, FILE_NAME = 'BERT-Base-Uncased', 'test', 'BERT-ML-Uncased-jigsaw_6lbls_BCE_128_CECE_testing'
with open(f'./Metrics_results/{MODEL_NAME}/{MODE}/{FILE_NAME}.pkl', 'rb') as file:
    data = pickle.load(file)
    
metrics = data[0]
required_metrics = dict(Model=MODEL_NAME, micro_precision = metrics['precision_micro'], 
                        macro_precision=metrics['precision_macro'], micro_recall = metrics['recall_micro'], macro_recall=metrics['recall_macro'], 
                        F1_per_class = metrics['f1_per_class'], auc_per_class=metrics['auc_per_class'])    
print('\n', MODEL_NAME, required_metrics)

MODEL_NAME, MODE, FILE_NAME = 'distilbert-base-cased', 'test', 'DistilBERT-ML-Cased-jigsaw_6lbls_BCE__training'
with open(f'./Metrics_results/{MODEL_NAME}/{MODE}/{FILE_NAME}.pkl', 'rb') as file:
    data = pickle.load(file)
    
metrics = data[0]
required_metrics = dict(Model=MODEL_NAME, micro_precision = metrics['precision_micro'], 
                        macro_precision=metrics['precision_macro'], micro_recall = metrics['recall_micro'], macro_recall=metrics['recall_macro'], 
                        F1_per_class = metrics['f1_per_class'], auc_per_class=metrics['auc_per_class'])    
print('\n', MODEL_NAME, required_metrics)

##DistilBERT-ML-Cased-jigsaw_6lbls_FL__training.pkl

MODEL_NAME, MODE, FILE_NAME = 'distilbert-base-uncased', 'test', 'DistilBERT-ML-Uncased-jigsaw_6lbls_BCE_128CECE_training'
with open(f'./Metrics_results/{MODEL_NAME}/{MODE}/{FILE_NAME}.pkl', 'rb') as file:
    data = pickle.load(file)
    
metrics = data[0]
required_metrics = dict(Model=MODEL_NAME, micro_precision = metrics['precision_micro'], 
                        macro_precision=metrics['precision_macro'], micro_recall = metrics['recall_micro'], macro_recall=metrics['recall_macro'], 
                        F1_per_class = metrics['f1_per_class'], auc_per_class=metrics['auc_per_class'])    
print('\n', MODEL_NAME, required_metrics)

MODEL_NAME, MODE, FILE_NAME = 'HateBert', 'test', 'HateBert-ML-jigsaw_6lbls_BCE_128_testing'
with open(f'./Metrics_results/{MODEL_NAME}/{MODE}/{FILE_NAME}.pkl', 'rb') as file:
    data = pickle.load(file)
    
metrics = data[0]
required_metrics = dict(Model=MODEL_NAME, micro_precision = metrics['precision_micro'], 
                        macro_precision=metrics['precision_macro'], micro_recall = metrics['recall_micro'], macro_recall=metrics['recall_macro'], 
                        F1_per_class = metrics['f1_per_class'], auc_per_class=metrics['auc_per_class'])    
print('\n', MODEL_NAME, required_metrics)

MODEL_NAME, MODE, FILE_NAME = 'roberta-base', 'test', 'roberta-ML-Cased-jigsaw_6lbls_BCE__training'
with open(f'./Metrics_results/{MODEL_NAME}/{MODE}/{FILE_NAME}.pkl', 'rb') as file:
    data = pickle.load(file)

metrics = data[0]
required_metrics = dict(Model=MODEL_NAME, micro_precision = metrics['precision_micro'], 
                        macro_precision=metrics['precision_macro'], micro_recall = metrics['recall_micro'], macro_recall=metrics['recall_macro'], 
                        F1_per_class = metrics['f1_per_class'], auc_per_class=metrics['auc_per_class'],
                        CECE = metrics['CECE']
                        )    
print('\n', MODEL_NAME, required_metrics)