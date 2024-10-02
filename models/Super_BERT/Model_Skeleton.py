import torch
from torch.utils.data import Dataset
from transformers import AutoModelForSequenceClassification, AutoConfig

class Model_Configuration :
    def __init__(self, batch_size, samples) -> None:
        self.batch_size = batch_size
        self.samples = samples
        
    def calc_step_size(self):
        if self.batch_size <= 0:
            raise ValueError("Batch size must be a positive integer")
        return self.samples // self.batch_size # integer division
    
    def load_model (self, model, lr, max_lr, weight_decay, step_size_up, mode='triangular2', useStep=False, gamma=.1) -> None:
        binary_optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = torch.optim.lr_scheduler.CyclicLR(binary_optimizer, base_lr=lr, max_lr=max_lr, step_size_up=step_size_up, mode=mode, cycle_momentum=True)
        if useStep :
            scheduler = torch.optim.lr_scheduler.StepLR(binary_optimizer, step_size=step_size_up, gamma=gamma)
        loss = torch.nn.BCEWithLogitsLoss().cuda()
        return binary_optimizer, scheduler, loss
    
    
    
    

class HateCrimeModel (torch.nn.Module):
    
    def __init__(self, model_name ='distilbert-base-uncased', num_labels = 7, batch_size=16, attn_dropout = .1 , hidden_dropout = .1, model_dropout = .1) -> None:
        super(HateCrimeModel, self).__init__()
        '''
            Part I : Tackling Blacklisting and White listing for hate comments
        '''
        binary_labels = 1
        self.batch_size = 16
        # Dropout customization for model =] 
        binary_config = AutoConfig.from_pretrained(model_name, num_labels=binary_labels)
        binary_config.hidden_dropout_prob = hidden_dropout
        binary_config.attention_probs_dropout_prob = attn_dropout
        
        self.HateCrimeBlackListDropout = torch.nn.Dropout(model_dropout)
        
        # activation
        self.model_relu = torch.nn.ReLU()
        
        # Model configuration and updates
        self.binary_classification_model = AutoModelForSequenceClassification.from_pretrained(model_name, config = binary_config)
        self.hidden_size = self.binary_classification_model.config.hidden_size
        
    def forward(self, ids, mask):
        ################### First Part : Black List or White List [0: White List, 1: Black List] ###############################
        output = self.binary_classification_model(input_ids = ids, attention_mask = mask)
        y = output.logits#.squeeze() # if something happens add .squeeze to remove that extra dimension =]
        y = self.HateCrimeBlackListDropout(y)
        return y
    
    
    
    ########################################################################################################################################################################
                                                                    #  Dataset
        
class BinaryHateCrimeDataset (Dataset):
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
            dictionary = dict(index=index, input_ids=input_ids, attention_mask=attn_mask, labels= labels[:-1], blacklist=labels[-1]) # <-- Black List or White List
        else:
            dictionary = dict(index=index, input_ids=input_ids, attention_mask=attn_mask)
        return dictionary
        
        
            