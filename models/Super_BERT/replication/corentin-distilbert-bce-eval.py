import torch 
from loguru import logger
import json
import tqdm 
import pandas as pd
import transformers 
from torch.utils.data import DataLoader, Dataset
from typing import Any, Union, Dict, List
from skeleton import TransformerClassifierStack
from torchmetrics import MetricCollection, Metric, Accuracy, Precision, Recall, AUROC, HammingDistance, F1Score, ROC, AUROC
import os
import warnings

tokenizer = transformers.AutoTokenizer.from_pretrained('distilbert-base-uncased')
BATCH_SIZE = 32

warnings.filterwarnings("ignore", message="No positive samples in targets, true positive value should be meaningless")

METRIC_FILE_PATH = '././Metrics_results/distilbert-base-uncased/distilbert_uncased-test.metric.json'

def serialize(object_to_serialize: Any, ensure_ascii: bool = True) -> str:
    """
    Serialize any object, i.e. convert an object to JSON
    Args:
        object_to_serialize (Any): The object to serialize
        ensure_ascii (bool, optional): If ensure_ascii is true (the default), the output is guaranteed to have all incoming non-ASCII characters escaped. If ensure_ascii is false, these characters will be output as-is. Defaults to True.
    Returns:
            str: string of serialized object (JSON)
    """

    def dumper(obj: Any) -> Union[str, Dict]:
        """
        Function called recursively by json.dumps to know how to serialize an object.
        For example, for datetime, we try to convert it to ISO format rather than
        retrieve the list of attributes defined in its object.
        Args:
            obj (Any): The object to serialize
        Returns:
            Union[str, Dict]: Serialized object
        """
        if isinstance(obj, torch.Tensor):
            return obj.cpu().numpy().tolist()
        elif hasattr(obj, "__dict__"):
            return obj.__dict__
        return str(obj)

    return json.dumps(object_to_serialize, default=dumper, ensure_ascii=ensure_ascii)

def export_metric(metric_collection, **kwargs):
    """
    Export MetricCollection to json file

    Args:
        metric_collection: MetricCollection
        **kwargs: field to add in json line
    """
    with open(METRIC_FILE_PATH, "a") as f:
        metric_collection_value = metric_collection.compute()
        metric_collection_value.update(kwargs)
        serialized_value = serialize(metric_collection_value)
        f.write(serialized_value)
        f.write("\n")
    logger.success("Metrics are exported !")


class JigsawDataset(Dataset):
    def __init__(self, data_df, tokenizer):
        self.data = data_df
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        comment = self.data.iloc[index]["comment_text"]
        label = torch.tensor(self.data.iloc[index][LABEL_LIST].tolist(), dtype=torch.float)
        
        token_list, attention_mask = self.text_to_token_and_mask(comment)

        return dict(index=index, ids=token_list, mask=attention_mask, labels=label)
    
    def text_to_token_and_mask(self, input_text):
        tokenization_dict = self.tokenizer.encode_plus(input_text,
                                add_special_tokens=True,
                                max_length=128,
                                padding='max_length',
                                truncation=True,
                                return_attention_mask=True,
                                return_tensors='pt')
        
        token_list = tokenization_dict["input_ids"].flatten()
        attention_mask = tokenization_dict["attention_mask"].flatten()
        return (token_list, attention_mask)

torch.cuda.empty_cache() 
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # load the device as cuda if you have a graphic card or cpu if not
num_labels = 6
LABEL_LIST = ['toxicity', 'obscene', 'sexual_explicit',
            'identity_attack', 'insult', 'threat']

test_df = pd.read_csv('././Data/jigsaw.test.csv')
test_dataset = JigsawDataset(test_df, tokenizer)
test_dataloader = DataLoader(test_dataset,
                             batch_size=BATCH_SIZE,
                             shuffle=False)


test_metric_dict = dict()

# AUROC Macro
auroc_macro = AUROC(num_labels=num_labels, task='multilabel', average="macro")
test_metric_dict["auroc_macro"] = auroc_macro

# AUROC per class
auroc_per_class = AUROC(num_labels=num_labels, task='multilabel', average=None)
test_metric_dict["auroc_per_class"] = auroc_per_class

# F1 score global
f1 = F1Score(num_labels=num_labels, average='micro', task='multilabel')
test_metric_dict["f1_micro"] = f1

f1 = F1Score(num_labels=num_labels, average='macro', task='multilabel')
test_metric_dict["f1_macro"] = f1

# F1 score per class
f1_per_calss = F1Score(num_labels=num_labels, task='multilabel', average=None)
test_metric_dict["f1_per_calss"] = f1_per_calss


precision = Precision(num_labels=num_labels, task='multilabel', average='macro')
test_metric_dict["precision_macro"] = precision

precision_micro = Precision(num_labels=num_labels, task='multilabel', average='micro')
test_metric_dict["precision_micro"] = precision_micro


recall = Recall(num_labels=num_labels, task='multilabel', average='macro')
test_metric_dict["recall_macro"] = recall

recall_micro = Recall(num_labels=num_labels, task='multilabel', average='micro')
test_metric_dict["recall_micro"] = recall_micro


test_metric = MetricCollection(test_metric_dict)
test_metric.to(device)

@torch.no_grad()
def evaluation(model):
    model.eval()
    logger.info(f"START EVALUATION")

    index_tensor = torch.Tensor([])
    prediction_tensor = torch.Tensor([])

    progress = tqdm.tqdm(test_dataloader, desc='test batch...', leave=False)
    for batch_id, batch in enumerate(progress):
        logger.trace(f"{batch_id=}")
        index_batch = batch["index"].to(device)
        token_list_batch = batch["ids"].to(device)
        attention_mask_batch = batch["mask"].to(device)
        label_batch = batch["labels"].to(device)

        # Predict
        prediction_batch = model(token_list_batch, attention_mask_batch)
        transformed_prediction_batch = prediction_batch.squeeze()
        proba_prediction_batch = torch.sigmoid(transformed_prediction_batch)
        
        test_metric(proba_prediction_batch.to(torch.float32), label_batch.to(torch.int32))
        
        #index_tensor = torch.concat([index_tensor, index_batch.cpu()])
        #prediction_tensor = torch.concat([prediction_tensor, proba_prediction_batch.cpu()])
    
    logger.info(f"END EVALUATION")
    # prediction_test_df = pd.DataFrame(prediction_tensor.tolist(), 
    #                                  columns=LABEL_LIST,
    #                                  index=index_tensor.to(int).tolist())
    
    export_metric(test_metric, epoch_id=1, batch_id=batch_id, loss=None)
    
    #prediction_test_df.to_csv('./test-bert-bce-uncased.csv')
    logger.success(f"Test predictions exported !")
    
    
tr_model = transformers.AutoModel.from_pretrained('distilbert-base-uncased')
#model = TransformerClassifierStack(tr_model, num_labels, freeze=True) 
model = torch.load("././saved/distilbert-base-uncased/distilbert-base-uncased-corentin.model")

evaluation(model)