import torch 
from torch.utils.data import Dataset
from transformers import AutoModelForSequenceClassification


class Model_Skeleton (torch.nn.Module) : 
    
    def __init__(self, num_labels = 7, model_name='bert-base-cased'):
        
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.hidden_dim = self.model.config.hidden_size 
        
        self.num_labels = num_labels
        
    def forward(self, ids, mask) :
        model_output = self.model(input_ids = ids, attention_mask = mask)#, output_hidden_states=True)
        print (model_output)
        


############################### Dataset ######################################

class HateSpeechDataset (Dataset) :
    
    def __init__(self, dataset) -> None:
        super().__init__()
        self.data = dataset
        self.has_labels = 'labels' in self.data
        
    def __len__(self):
        return self.data['input_ids'].size(0)
    
    
    def __getitem__(self, index):
        dictionary = {}
        input_ids, attn_mask = self.data['input_ids'][index], self.data['attention_mask'][index]
        if self.has_labels:
            labels = self.data['labels'][index]
            dictionary = dict(index=index, input_ids=input_ids, attention_mask=attn_mask, labels=labels)
        else:
            dictionary = dict(index=index, input_ids=input_ids, attention_mask=attn_mask)
        return dictionary
    