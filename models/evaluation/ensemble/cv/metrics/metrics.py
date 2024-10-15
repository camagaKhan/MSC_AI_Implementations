import pickle as pkl
 
path = f'././././Metrics_results/ensemble/distilbert_cased-ML-Cased-jigsaw_6lbls_{'FL'}_ensemble_averaging_training.pkl'
with open(path, 'rb') as file:
    metrics = pkl.load(file)
    #auc , macro_f1, micro f1, weighted f1, Precision, Recall, CECE
    my_metrics = dict(AUC_MACRO = metrics['results']['auc_roc_macro'],
                      AUC_PER_CLASS = metrics['results']['auc_per_class'],
                      F1_PER_CLASS = metrics['results']['f1_per_class'],
                      F1_Macro = metrics['results']['f1_Macro'], 
                      F1_Micro = metrics['results']['f1_Micro'], 
                      F1_Weighted = metrics['results']['f1_Weighted'],
                      Precision_Macro = metrics['results']['precision_macro'],
                      Precision_Micro = metrics['results']['precision_micro'],
                      Recall_Macro = metrics['results']['recall_macro'],
                      Recall_Micro = metrics['results']['recall_micro'],
                      CECE = metrics['cece'])
    print('\ndistilbert ensemble\n', my_metrics)
    
    
path = f'././././Metrics_results/ensemble/bert_cased-ML-Cased-jigsaw_6lbls_{'FL'}_ensemble_averaging_training.pkl'
with open(path, 'rb') as file:
    metrics = pkl.load(file)
    #auc , macro_f1, micro f1, weighted f1, Precision, Recall, CECE
    my_metrics = dict(AUC_MACRO = metrics['results']['auc_roc_macro'],
                      AUC_PER_CLASS = metrics['results']['auc_per_class'],
                      F1_PER_CLASS = metrics['results']['f1_per_class'],
                      F1_Macro = metrics['results']['f1_Macro'], 
                      F1_Micro = metrics['results']['f1_Micro'], 
                      F1_Weighted = metrics['results']['f1_Weighted'],
                      Precision_Macro = metrics['results']['precision_macro'],
                      Precision_Micro = metrics['results']['precision_micro'],
                      Recall_Macro = metrics['results']['recall_macro'],
                      Recall_Micro = metrics['results']['recall_micro'],
                      CECE = metrics['cece'])
    print('\n bert-cased ensemble\n', my_metrics)
    
    
path = f'././././Metrics_results/ensemble/roberta_base-ML-Cased-jigsaw_6lbls_{'FL'}_ensemble_averaging_training.pkl'
with open(path, 'rb') as file:
    metrics = pkl.load(file)
    #auc , macro_f1, micro f1, weighted f1, Precision, Recall, CECE
    my_metrics = dict(AUC_MACRO = metrics['results']['auc_roc_macro'],
                      AUC_PER_CLASS = metrics['results']['auc_per_class'],
                      F1_PER_CLASS = metrics['results']['f1_per_class'],
                      F1_Macro = metrics['results']['f1_Macro'], 
                      F1_Micro = metrics['results']['f1_Micro'], 
                      F1_Weighted = metrics['results']['f1_Weighted'],
                      Precision_Macro = metrics['results']['precision_macro'],
                      Precision_Micro = metrics['results']['precision_micro'],
                      Recall_Macro = metrics['results']['recall_macro'],
                      Recall_Micro = metrics['results']['recall_micro'],
                      CECE = metrics['cece'])
    print('\n roberta-base ensemble\n', my_metrics)