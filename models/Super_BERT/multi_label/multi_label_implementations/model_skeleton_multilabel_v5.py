import torch 
from torch.utils.data import Dataset
import torch.nn as nn 
import torch.nn.functional as F
from transformers import AutoModel, AutoModelForSequenceClassification, AutoConfig, AutoTokenizer

class HateSpeechv2Dataset (Dataset):
    def __init__(self, dataset, model_name, max_length = 280, multi_label = True, forTraining = True, without_sexual_explict = True) -> None:
        super().__init__()
        self.data = dataset
        print(len(self.data))
        self.has_labels = forTraining
        self.model_name = model_name
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        if multi_label: 
            if without_sexual_explict: 
                self.targets = ['toxicity', 'obscene', 'identity_attack', 'insult', 'threat'] # sexual_explicit is the smallest column after severe_toxicity. I removed it to see if classification improves
            else :
                self.targets = ['toxicity', 'obscene', 'sexual_explicit', 'identity_attack', 'insult', 'threat']
        
        
    def __len__(self):
        get_length = len(self.data['comment_text'])
        return get_length#.size(0)
    
    def __getitem__(self, index):
        dictionary, item = {}, self.data.iloc[index]
        tokens = self.tokenizer(item['comment_text'], add_special_tokens=True, max_length=self.max_length, padding='max_length', truncation = True, return_tensors='pt')
        if self.has_labels :
            labels = torch.tensor(item[self.targets].astype(float).values, dtype=torch.float)
            dictionary = dict(index=index, input_ids=tokens['input_ids'].squeeze(), attention_mask=tokens['attention_mask'].squeeze(), labels=labels)
        else:
            dictionary = dict(index=index, input_ids=tokens['input_ids'], attention_mask=tokens['attention_mask'])
        return dictionary

class HateSpeechDataset (Dataset):
    def __init__(self, dataset):
        self.data = dataset
        print(self.data['input_ids'].shape)
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


class HateSpeechTagger(nn.Module) :    
    
    def __init__(self, model_name, n_classes=6, model_dropout=.3, hidden_dropout=.3, attn_dropout=.3) -> None:
        super().__init__()
        
        self.model_name = model_name
        self.n_classes = n_classes
        
        trans_config = AutoModelForSequenceClassification.from_pretrained(self.model_name, num_labels=self.n_classes)
        trans_config.hidden_dropout_prob = hidden_dropout
        trans_config.attention_probs_dropout_prob = attn_dropout
        
        # initializing the transformer for classification of hate speech data
        self.transformer = AutoModel.from_pretrained(self.model_name, config=trans_config)
        
        self.hidden_dim = self.transformer.config.hidden_size * 4 # <-- we're using the last four hidden states for this experiment        
        self.dropout = nn.Dropout(model_dropout)
        
        self.dense = nn.Linear(self.hidden_dim, self.hidden_dim) # <-- We can learn from these embeddings. First test on last_hidden_state
        # final layer... probabilities will pass through the linear layer.
        self.classify = nn.Linear(self.hidden_dim, self.n_classes)
        
        self.batch_norm = nn.BatchNorm1d(self.hidden_dim)
        
        # set the weights to use kaiming_uniform
        torch.nn.init.kaiming_uniform_(self.classify.weight)
        torch.nn.init.kaiming_uniform_(self.dense.weight)
        
        if self.classify.bias is not None:
            torch.nn.init.zeros_(self.classify.bias)
        
        # since we are not using non-linearity like RELU there is no need for dropout before the forward. I will add dropout after first test
        
    def forward (self, input_ids, attention_mask):
        
        output = self.transformer(input_ids, attention_mask=attention_mask, output_hidden_states=True)
        hidden_states = output['hidden_states']
        
        #all_hidden_states = torch.stack(hidden_states, dim=0) # Shape: (num_layers, batch_size, seq_len, hidden_size)
        #max_pooled_states, _ = torch.max(all_hidden_states, dim=0)  # Shape: (batch_size, seq_len, hidden_size)
        #features = max_pooled_states[:, 0, :] # <-- this is the cls representation
        
        features = torch.cat([
            hidden_states[-1][:,0,:], 
            hidden_states[-2][:,0,:],
            hidden_states[-3][:,0,:],
            hidden_states[-4][:,0,:]
        ], dim=-1) # corentin duchene et al. do this.
        
        #features = torch.mean(torch.stack(hidden_states), dim=0)[:, 0, :] # I'm using all hidden states
        
        # find patterns in the data... 
        output = self.dropout(features)
        output = self.dense(output)
        #output = self.batch_norm(output)
        output = F.relu(output)        
        
        # get probabilities
        output = self.dropout(output)
        logits = self.classify(output) # last hidden state is used for classification.         
          
        return logits  # <-- returns size [batch, no_labels]