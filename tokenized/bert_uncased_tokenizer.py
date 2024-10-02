from transformers import AutoTokenizer
import pandas as pd
import torch

print('Reading datasets from csv')
train = pd.read_csv('./Data/jigsaw.train.csv')
validation = pd.read_csv('./Data/jigsaw.validation.csv')
test = pd.read_csv('./Data/jigsaw.test.csv')

targets = ['toxicity', 'obscene', 'sexual_explicit', 'identity_attack', 'insult', 'threat', 'not_offensive']

model_name = 'bert-base-uncased'
tokenizer = AutoTokenizer.from_pretrained(model_name)
MAX_LENGTH = 200

print('Starting Tokenization...')

train_tokenized = tokenizer(list(train['comment_text']), max_length=MAX_LENGTH, padding='max_length', truncation = True)
test_tokenized = tokenizer(list(test['comment_text']), max_length=MAX_LENGTH, padding='max_length', truncation = True)
validation_tokenized = tokenizer(list(validation['comment_text']), max_length=MAX_LENGTH, padding='max_length', truncation = True)

print('Finished Tokenization...')

train_dict = {
    'input_ids' : torch.tensor(train_tokenized['input_ids']), 
    'attention_mask': torch.tensor(train_tokenized['attention_mask']),
    'labels' : torch.tensor(train[targets].values, dtype=torch.float)
}


test_dict = {
    'input_ids' : torch.tensor(test_tokenized['input_ids']), 
    'attention_mask': torch.tensor(test_tokenized['attention_mask']),
    'labels' : torch.tensor(test[targets].values, dtype=torch.float)
}


validation_dict = {
    'input_ids' : torch.tensor(validation_tokenized['input_ids']), 
    'attention_mask': torch.tensor(validation_tokenized['attention_mask']),
    'labels' : torch.tensor(validation[targets].values, dtype=torch.float)
}

print('Saving dictionaries...')

torch.save(train_dict, './tokenized/BERT-BASE-UNCASED/train.pth')
torch.save(test_dict, './tokenized/BERT-BASE-UNCASED/test.pth')
torch.save(validation_dict, './tokenized/BERT-BASE-UNCASED/validation.pth')

print('all done')