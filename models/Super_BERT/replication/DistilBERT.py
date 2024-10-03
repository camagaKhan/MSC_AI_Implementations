import torch 
from skeleton import TransformerClassifierStack
import transformers 
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import numpy as np
import tqdm
import json
from typing import Any, Union, Dict, List
from loguru import logger
from torchmetrics import MetricCollection, Metric, Accuracy, Precision, Recall, AUROC, HammingDistance, F1Score, ROC, AUROC
from sklearn.model_selection import train_test_split

import warnings

# Suppress the specific warning related to no positive samples in targets
warnings.filterwarnings("ignore", message="No positive samples in targets, true positive value should be meaningless")



nb_labels = 6
LABEL_LIST = ['toxicity', 'obscene', 'sexual_explicit',
            'identity_attack', 'insult', 'threat']

tokenizer = transformers.AutoTokenizer.from_pretrained('distilbert-base-uncased')
tr_model = transformers.AutoModel.from_pretrained('distilbert-base-uncased')
model = TransformerClassifierStack(tr_model, nb_labels, freeze=True) # Duchene freezes the layer

BATCH_SIZE = 32
LR=1e-4
PIN_MEMORY = True
NUM_WORKERS = 0
PREFETCH_FACTOR = None
NUM_EPOCHS = 1

torch.cuda.empty_cache() 
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # load the device as cuda if you have a graphic card or cpu if not

METRIC_FILE_PATH = './././Metric_results/distilbert-base-uncased/distilbert_uncased.metric.json'

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
        tokenization_dict = tokenizer.encode_plus(input_text,
                                add_special_tokens=True,
                                max_length=128,
                                padding='max_length',
                                truncation=True,
                                return_attention_mask=True,
                                return_tensors='pt')
        token_list = tokenization_dict["input_ids"].flatten()
        attention_mask = tokenization_dict["attention_mask"].flatten()
        return (token_list, attention_mask)

#train_df, validation_df = pd.read_csv('././Data/jigsaw.15.train.multi-label.csv'), pd.read_csv('././Data/jigsaw.validation.multi-label_mixed.csv')
all_data = pd.read_csv('././Data/all_data_cleaned.csv')
train_df, validation_df = train_test_split(all_data, test_size=0.2, random_state=42)
del all_data

train_df[LABEL_LIST] = (train_df[LABEL_LIST]>=0.5).astype(int)
validation_df[LABEL_LIST] = (validation_df[LABEL_LIST]>=0.5).astype(int)

train_df_0 = train_df[(train_df['toxicity'] == 0) &
                      (train_df['obscene'] == 0) &
                      (train_df['identity_attack'] == 0) &
                      (train_df['insult'] == 0) &
                      (train_df['threat'] == 0) &
                      (train_df['sexual_explicit'] == 0)]

train_df_1 = train_df[(train_df['toxicity'] == 1) |
                      (train_df['obscene'] == 1) |
                      (train_df['identity_attack'] == 1) |
                      (train_df['insult'] == 1) |
                      (train_df['threat'] == 1) |
                      (train_df['sexual_explicit'] == 1)]

nb_0 = len(train_df_0)
n_sampling = 0.1
nb_1 = int(nb_0 * n_sampling)
print("NB 0: {}".format(nb_0))
print("NB 1: {}".format(nb_1))
ids_0 = np.random.randint(0, high=nb_0, size=nb_1)

train_df_0 = train_df_0.iloc[ids_0]
train_df_2 = pd.concat([train_df_0, train_df_1])

print("Train size: {}".format(len(train_df_2)))

train_df = train_df_2

train_dataset = JigsawDataset(train_df, tokenizer)
train_dataloader = DataLoader(train_dataset,
                             batch_size=BATCH_SIZE,
                             shuffle=True,
                             num_workers=NUM_WORKERS,
                             prefetch_factor=PREFETCH_FACTOR,
                             pin_memory=PIN_MEMORY)


validation_dataset = JigsawDataset(validation_df, tokenizer)
validation_dataloader = DataLoader(validation_dataset,
                             batch_size=BATCH_SIZE,
                             shuffle=True,
                             num_workers=NUM_WORKERS,
                             prefetch_factor=PREFETCH_FACTOR,
                             pin_memory=PIN_MEMORY)

# Pas besoin de Sigmoid en sorti du model seulement pour `BCEWithLogitsLoss`
criterion = torch.nn.BCEWithLogitsLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

model.to(device)
criterion.to(device)


num_labels = len(LABEL_LIST)
train_metric_dict = dict()

# AUROC Macro
auroc_macro = AUROC(num_labels=num_labels, task='multilabel', average="macro")
train_metric_dict["auroc_macro"] = auroc_macro

# AUROC per class
auroc_per_class = AUROC(num_labels=num_labels, task='multilabel', average=None)
train_metric_dict["auroc_per_class"] = auroc_per_class

# F1 score global
f1 = F1Score(num_labels=num_labels, task='multilabel')
train_metric_dict["f1"] = f1

# F1 score per class
f1_per_calss = F1Score(num_labels=num_labels, task='multilabel', average=None)
train_metric_dict["f1_per_calss"] = f1_per_calss


precision = Precision(num_labels=num_labels, task='multilabel', average='macro')
train_metric_dict["precision_macro"] = precision

precision_micro = Precision(num_labels=num_labels, task='multilabel', average='micro')
train_metric_dict["precision_micro"] = precision_micro


recall = Recall(num_labels=num_labels, task='multilabel', average='macro')
train_metric_dict["recall_macro"] = recall

recall_micro = Recall(num_labels=num_labels, task='multilabel', average='micro')
train_metric_dict["recall_micro"] = recall_micro


train_metric = MetricCollection(train_metric_dict)
train_metric.to(device)

validation_metric = train_metric.clone()
validation_metric.to(device)

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


def train_epoch(epoch_id=None):
    model.train()
    logger.info(f"START EPOCH {epoch_id=}")

    progress = tqdm.tqdm(train_dataloader, desc='training batch...', leave=False)
    for batch_id, batch in enumerate(progress):
        if batch_id % 1_000 == 0:
            valid_epoch(epoch_id=epoch, batch_id=batch_id)
        
        logger.trace(f"{batch_id=}")
        token_list_batch = batch["ids"].to(device)
        attention_mask_batch = batch["mask"].to(device)
        label_batch = batch["labels"].to(device)

        # Reset gradient
        optimizer.zero_grad()

        # Predict
        prediction_batch = model(token_list_batch, attention_mask_batch)
        transformed_prediction_batch = prediction_batch.squeeze()

        # Loss
        loss = criterion(transformed_prediction_batch.to(torch.float32), label_batch.to(torch.float32))

        # Metrics
        proba_prediction_batch = torch.sigmoid(transformed_prediction_batch)
        train_metrics_collection_dict = train_metric(proba_prediction_batch.to(torch.float32), label_batch.to(torch.int32))
        logger.trace(train_metrics_collection_dict)

        # Backprop        
        loss.backward()
        # gradient clip
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        # Update progress bar description
        progress_description = "Train Loss : {loss:.4f} - Train AUROC : {acc:.4f}"
        auroc_macro_value = float(train_metrics_collection_dict["auroc_macro"])
        progress_description = progress_description.format(loss=loss.item(), acc=auroc_macro_value)
        progress.set_description(progress_description)

    logger.info(f"END EPOCH {epoch_id=}")
    
@torch.no_grad()
def valid_epoch(epoch_id=None, batch_id=None):
    model.eval()
    logger.info(f"START VALIDATION {epoch_id=}{batch_id=}")
    validation_metric.reset()

    loss_list = []
    prediction_list = torch.Tensor([])
    target_list = torch.Tensor([])


    progress = tqdm.tqdm(validation_dataloader, desc="valid batch...", leave=False)
    for _, batch in enumerate(progress):
        
        token_list_batch = batch["ids"].to(device)
        attention_mask_batch = batch["mask"].to(device)
        label_batch = batch["labels"].to(device)

        # Predict
        prediction_batch = model(token_list_batch, attention_mask_batch)

        transformed_prediction_batch = prediction_batch.squeeze()

        # Loss
        loss = criterion(
            transformed_prediction_batch.to(torch.float32),
            label_batch.to(torch.float32),
        )

        loss_list.append(loss.item())

        proba_prediction_batch = torch.sigmoid(transformed_prediction_batch)
        prediction_list = torch.concat(
            [prediction_list, proba_prediction_batch.cpu()]
        )
        target_list = torch.concat([target_list, label_batch.cpu()])

        # Metrics
        preds, labels = proba_prediction_batch.shape, label_batch.shape
        validation_metric(proba_prediction_batch.to(torch.float32), label_batch.to(torch.int32))

    loss_mean = np.mean(loss_list)
    logger.trace(validation_metric.compute())
    logger.info(f"END VALIDATION {epoch_id=}{batch_id=}")
    export_metric(validation_metric, epoch_id=epoch_id, batch_id=batch_id, loss=loss_mean)
    
torch.cuda.empty_cache()
progress =  tqdm.tqdm(range(1,NUM_EPOCHS+1), desc='training epoch...', leave=True)
MODEL_FILE_PATH = f'./././saved/distilbert-base-uncased/distilbert-base-uncased-corentin.model'
for epoch in progress:
    # Train
    train_epoch(epoch_id=epoch)

    # Validation
    valid_epoch(epoch_id=epoch)

    # Save
    torch.save(model, MODEL_FILE_PATH)

