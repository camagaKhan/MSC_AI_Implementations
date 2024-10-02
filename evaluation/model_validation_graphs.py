import glob 
import pickle as pkl 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from matplotlib import rcParams
rcParams['font.family'] = 'Arial'


folder = 'bert-base-cased' 
directory = f'.\\Metrics_results\\{folder}\\validation\\binary\\{folder}_validation_*.pkl'
file_list, bert_cased_validation_data = glob.glob(directory), []

for pkl_file in file_list:
    with open(pkl_file, 'rb') as file:
        bert_cased_validation_data.append(pkl.load(file))
        
bert_base_cased_val_loss = []
for validation_iter_results in bert_cased_validation_data:
    for result_in_epoch in validation_iter_results:
        bert_base_cased_val_loss.append(dict(epoch= result_in_epoch['epoch'], loss = result_in_epoch['loss']))
        
print(folder, bert_base_cased_val_loss)


#############################################################################################################################

folder = 'BERT-Base-Uncased' 
directory = f'.\\Metrics_results\\{folder}\\validation\\binary\\{folder}_validation_*.pkl'
file_list, bert_uncased_validation_data = glob.glob(directory), []

for pkl_file in file_list:
    with open(pkl_file, 'rb') as file:
        bert_uncased_validation_data.append(pkl.load(file))
        
bert_base_uncased_val_loss = []
for validation_iter_results in bert_uncased_validation_data:
    for result_in_epoch in validation_iter_results:
        bert_base_uncased_val_loss.append(dict(epoch= result_in_epoch['epoch'], loss = result_in_epoch['loss']))
        
print(folder, bert_base_uncased_val_loss)


#############################################################################################################################

folder = 'mBERTu' 
directory = f'.\\Metrics_results\\{folder}\\validation\\{folder}_validation_*.pkl'
file_list, mbertu_validation_data = glob.glob(directory), []

for pkl_file in file_list:
    with open(pkl_file, 'rb') as file:
        mbertu_validation_data.append(pkl.load(file))
        
mbertu_val_loss = []
for validation_iter_results in mbertu_validation_data:
    for result_in_epoch in validation_iter_results:
        mbertu_val_loss.append(dict(epoch= result_in_epoch['epoch'], loss = result_in_epoch['loss']))
        
print(folder, mbertu_val_loss)


#############################################################################################################################

folder = 'roberta-base' 
directory = f'.\\Metrics_results\\{folder}\\validation\\binary\\{folder}_validation_*.pkl'
file_list, roberta_validation_data = glob.glob(directory), []

for pkl_file in file_list:
    with open(pkl_file, 'rb') as file:
        roberta_validation_data.append(pkl.load(file))
        
roberta_val_loss = []
for validation_iter_results in roberta_validation_data:
    for result_in_epoch in validation_iter_results:
        roberta_val_loss.append(dict(epoch= result_in_epoch['epoch'], loss = result_in_epoch['loss']))
        
print(folder, roberta_val_loss)


#############################################################################################################################

folder = 'GroNLP\\hateBERT' 
directory = f'.\\Metrics_results\\{folder}\\validation\\GroNLP\\{'hateBERT'}_validation_*.pkl'
file_list, hatebert_validation_data = glob.glob(directory), []

for pkl_file in file_list:
    with open(pkl_file, 'rb') as file:
        hatebert_validation_data.append(pkl.load(file))
        
hatebert_val_loss = []
for validation_iter_results in hatebert_validation_data:
    for result_in_epoch in validation_iter_results:
        hatebert_val_loss.append(dict(epoch= result_in_epoch['epoch'], loss = result_in_epoch['loss']))
        
print(folder, hatebert_val_loss)


#############################################################################################################################

folder = 'distilbert-base-cased' 
directory = f'.\\Metrics_results\\{folder}\\validation\\{folder}_validation_*.pkl'
file_list, distilbert_cased_validation_data = glob.glob(directory), []

for pkl_file in file_list:
    with open(pkl_file, 'rb') as file:
        distilbert_cased_validation_data.append(pkl.load(file))
        
distilbert_cased_val_loss = []
for validation_iter_results in distilbert_cased_validation_data:
    for result_in_epoch in validation_iter_results:
        distilbert_cased_val_loss.append(dict(epoch= result_in_epoch['epoch'], loss = result_in_epoch['loss']))
        
print(folder, distilbert_cased_val_loss)


#############################################################################################################################

folder = 'distilbert-base-uncased' 
directory = f'.\\Metrics_results\\{folder}\\validation\\{folder}_validation_*.pkl'
file_list, distilbert_uncased_validation_data = glob.glob(directory), []

for pkl_file in file_list:
    with open(pkl_file, 'rb') as file:
        distilbert_uncased_validation_data.append(pkl.load(file))
        
distilbert_uncased_val_loss = []
for validation_iter_results in distilbert_uncased_validation_data:
    for result_in_epoch in validation_iter_results:
        distilbert_uncased_val_loss.append(dict(epoch= result_in_epoch['epoch'], loss = result_in_epoch['loss']))
        
print(folder, distilbert_uncased_val_loss)


#############################################################################################################################

def get_last_losses_per_epoch(model_data):
    last_losses = {}
    for record in model_data:
        epoch = record['epoch']
        last_losses[epoch] = record['loss']
    return last_losses
    
    # last_losses = []
    # for record in model_data:
    #     #epoch = record['epoch']
    #     last_losses.append(record['loss'])
    # return last_losses

# Apply the function to each model
models = {
    "bert-base-cased": bert_base_cased_val_loss,
    "bert-base-uncased": bert_base_uncased_val_loss,
    "mBERTu": mbertu_val_loss,
    "roberta-base": roberta_val_loss,
    "GroNLP-hateBERT": hatebert_val_loss,
    "distilbert-base-cased": distilbert_cased_val_loss,
    "distilbert-base-uncased": distilbert_uncased_val_loss
}

data_list = []
epochs = [1, 2, 3]  # Assuming you have 3 epochs
for model_name, model_data in models.items():
    last_losses = get_last_losses_per_epoch(model_data)
    for epoch, loss in last_losses.items():
        data_list.append({"Epoch": epoch, "Loss": loss, "Model": model_name})

# Convert to a pandas DataFrame
df = pd.DataFrame(data_list)

# Plot the graph using seaborn
plt.figure(figsize=(8, 6))
sns.set_theme(style="whitegrid")

# Create a seaborn line plot
sns.lineplot(x='Epoch', y='Loss', hue='Model', marker='o', data=df)

# Adding labels and title
plt.xlabel('Epoch', fontsize=14)
plt.ylabel('Loss', fontsize=14)
plt.title('Loss vs Epoch', fontsize=16)
plt.grid(False)
plt.legend(loc='upper right', bbox_to_anchor=(1, 0.90))

plt.show()

print()
    
    #print(f"Last loss for each epoch in {model_name}: {last_losses}")