# -*- coding: utf-8 -*-
"""
Created on Tue Jun 20 12:09:14 2023

@author: Rajeswari
"""

import os
import pandas as pd
import torch
from tqdm.notebook import tqdm
from torch.utils.data import Dataset, DataLoader
#from ml_things import plot_dict, plot_confusion_matrix#, fix_text
from sklearn.metrics import classification_report, accuracy_score
import pickle
from transformers import set_seed, GPT2Config, GPT2Tokenizer, AdamW, get_linear_schedule_with_warmup, GPT2ForSequenceClassification
import json
import traceback

# import tracemalloc
# tracemalloc.start()

# Set seed for reproducibility.
set_seed(123)
epochs = 4
batch_size = 2
max_length = 60
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_name_or_path = 'gpt2'
labels_ids = {"Field_Declaration":0, "Field_Assignment":1}
n_labels = len(labels_ids)

#filepath = './Input_data/train_data.csv'
#model_name = 'gpt2'
#version = 1
#print(os.listdir())
#with open("./aidap/ml/properties.json", "r") as f:
with open("../properties.json", "r") as f:
	data = json.load(f)

model_path = data["classification"]['model_path']

class CustomTextDataset(Dataset):

    def __init__(self, path):
        df = pd.read_csv(path)
        self.labels = df['Type']
        self.text = df['Code']

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        label = self.labels[idx]
        data = self.text[idx]
        sample = {"Code": data, "Type": label}
        return sample
   

class Gpt2ClassificationCollator(object):

    def __init__(self, use_tokenizer, labels_encoder, max_sequence_len=None):

        self.use_tokenizer = use_tokenizer
        self.max_sequence_len = use_tokenizer.model_max_length if max_sequence_len is None else max_sequence_len
        self.labels_encoder = labels_encoder

        return

    def __call__(self, sequences):
        texts = [sequence['Code'] for sequence in sequences]
        # Get all labels from sequences list.
        labels = [sequence['Type'] for sequence in sequences]
        # Encode all labels using label encoder.
        labels = [self.labels_encoder[label] for label in labels]
        # Call tokenizer on all texts to convert into tensors of numbers with 
        # appropriate padding.
        inputs = self.use_tokenizer(text=texts, return_tensors="pt", padding=True, truncation=True, max_length=self.max_sequence_len)
        # Update the inputs with the associated encoded labels as tensor.
        inputs.update({'labels':torch.tensor(labels)})

        return inputs


async def train(model, dataloader, optimizer_, scheduler_, device_):
    # global model
    predictions_labels = []
    true_labels = []
    total_loss = 0
    model.train()
    # For each batch of training data...
    for batch in tqdm(dataloader, total=len(dataloader)):
        # Add original labels - use later for evaluation.
        true_labels += batch['labels'].numpy().flatten().tolist()
        
        # move batch to device
        batch = {k:v.type(torch.long).to(device_) for k, v in batch.items()}
        #print(batch)
        # Always clear any previously calculated gradients before performing a
        # backward pass.
        model.zero_grad()
        
        outputs = model(**batch)
        
        loss, logits = outputs[:2]
        
        total_loss += loss.item()
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        
        optimizer_.step()
        
        scheduler_.step()
        
        # Move logits and labels to CPU
        logits = logits.detach().cpu().numpy()
        
        # Convert these logits to list of predicted labels values.
        predictions_labels += logits.argmax(axis=-1).flatten().tolist()

    # Calculate the average loss over the training data.
    avg_epoch_loss = total_loss / len(dataloader)
  
    # Return all true labels and prediction for future evaluations.
    return true_labels, predictions_labels, avg_epoch_loss


async def validation(model, dataloader, device_):
    # Use global variable for model.
    # global model
    # Tracking variables
    predictions_labels = []
    true_labels = []
    # total loss for this epoch.
    total_loss = 0
    # Put the model in evaluation mode--the dropout layers behave differently
    # during evaluation.
    model.eval()
    # Evaluate data for one epoch
    for batch in tqdm(dataloader, total=len(dataloader)):
        # add original labels
        true_labels += batch['labels'].numpy().flatten().tolist()
        # move batch to device
        batch = {k:v.type(torch.long).to(device_) for k, v in batch.items()}
        # Telling the model not to compute or store gradients, saving memory and
        # speeding up validation
        with torch.no_grad(): 
            outputs = model(**batch)
            loss, logits = outputs[:2]
            logits = logits.detach().cpu().numpy()
            total_loss += loss.item()
            # get predicitons to list
            predict_content = logits.argmax(axis=-1).flatten().tolist()
            # update list
            predictions_labels += predict_content

    # Calculate the average loss over the training data.
    avg_epoch_loss = total_loss / len(dataloader)
    # Return all true labels and prediciton for future evaluations.
    return true_labels, predictions_labels, avg_epoch_loss


# Get model configuration.
print('Loading configuraiton...')
model_config = GPT2Config.from_pretrained(pretrained_model_name_or_path=model_name_or_path, num_labels=n_labels)

# Get model's tokenizer.
print('Loading tokenizer...')
tokenizer = GPT2Tokenizer.from_pretrained(pretrained_model_name_or_path=model_name_or_path)
# default to left padding
tokenizer.padding_side = "left"
# Define PAD Token = EOS Token = 50256
tokenizer.pad_token = tokenizer.eos_token

# Create data collator to encode text and labels into numbers.
gpt2_classificaiton_collator = Gpt2ClassificationCollator(use_tokenizer=tokenizer,
                                                          labels_encoder=labels_ids,
                                                          max_sequence_len=max_length)


async def trainModel(filepath, model_name, version):
    try:
    	with open("../properties.json","r") as f:
    		data = json.load(f)
    		
    	mp = data["classification"]["model_path"]
    	model_path = mp + f'{model_name}_v{version}.pkl'
    	# Get the actual model.
    	print('Loading model...')
    	if os.path.exists(model_path):
    		print('model exists')
    		model = pickle.load(open(model_path, 'rb'))
    		version = int(version) + 1
    	else:
    		print('model not found. creating new model...')
    		model = GPT2ForSequenceClassification.from_pretrained(pretrained_model_name_or_path=model_name_or_path, config=model_config)
    	model.resize_token_embeddings(len(tokenizer))
    	# fix model padding token id
    	model.config.pad_token_id = model.config.eos_token_id
    	# Load model to defined device.
    	model.to(device)
    	print('Model loaded to `%s`' % device)
    	print('Dealing with Train...')
    	# Create pytorch dataset.
    	train_dataset = CustomTextDataset(filepath)
    	print('Created `train_dataset` with %d examples!' % len(train_dataset))
    	
    	# Move pytorch dataset into dataloader.
    	train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=gpt2_classificaiton_collator)
    	print('Created `train_dataloader` with %d batches!' % len(train_dataloader))
    	print('Dealing with Validation...')
    	# Create pytorch dataset.
    	valid_dataset = CustomTextDataset(data["classification"]['val_data_path'])
    	print('Created `valid_dataset` with %d examples!' % len(valid_dataset))
    	# Move pytorch dataset into dataloader.
    	valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, collate_fn=gpt2_classificaiton_collator)
    	print('Created `eval_dataloader` with %d batches!' % len(valid_dataloader))
    	
    	optimizer = AdamW(model.parameters(),
                          lr=2e-5,  # default is 5e-5, our notebook had 2e-5
                          eps=1e-8  # default is 1e-8.
                          )
    	
    	total_steps = len(train_dataloader) * epochs
    	
    	scheduler = get_linear_schedule_with_warmup(optimizer,
                                                    num_warmup_steps=0,  # Default value in run_glue.py
                                                    num_training_steps=total_steps)
    	
    	all_loss = {'train_loss':[], 'val_loss':[]}
    	all_acc = {'train_acc':[], 'val_acc':[]}
    	
    	print('Epoch')
    	for epoch in tqdm(range(epochs)):
    		print(epoch)
    		print('Training on batches...')
    		# Perform one full pass over the training set.
    		train_labels, train_predict, train_loss = await train(model, train_dataloader, optimizer, scheduler, device)
    		train_acc = accuracy_score(train_labels, train_predict)
    		
    		print('Validation on batches...')
    		valid_labels, valid_predict, val_loss = await validation(model, valid_dataloader, device)
    		val_acc = accuracy_score(valid_labels, valid_predict)
    		print("  train_loss: %.5f - val_loss: %.5f - train_acc: %.5f - valid_acc: %.5f" % (train_loss, val_loss, train_acc, val_acc))
    		all_loss['train_loss'].append(train_loss)
    		all_loss['val_loss'].append(val_loss)
    		all_acc['train_acc'].append(train_acc)
    		all_acc['val_acc'].append(val_acc)
    	pickle.dump(model, open(mp + f'{model_name}_v{version}.pkl', 'wb'))
    	true_labels, predictions_labels, avg_epoch_loss = await validation(model, valid_dataloader, device)
    	
    	# Create the evaluation report.
    	evaluation_report = classification_report(true_labels, predictions_labels, labels=list(labels_ids.values()), target_names=list(labels_ids.keys()))
    	# Show the evaluation report.
    	print(evaluation_report)
    	return "Training succesful!"
    except Exception as e:
    	print(e)
    	traceback.print_exc()
    	return f"An error has occured: {e}"
        
async def inference(query, model_name, version):
    try:
          with open("../properties.json", "r") as f:
               data = json.load(f)
          model_path = data["classification"]['model_path']+ f'{model_name}_v{version}.pkl' 
          model = pickle.load(open(model_path, "rb"))
          model.to(device)
          inputs = tokenizer(query, return_tensors="pt")
          inputs.to(device)
          outputs = model(**inputs)
          logits = outputs.logits
          logits = logits.detach().cpu().numpy()
          predictions = logits.argmax(axis=-1).flatten().tolist()
          for i in predictions:
            predictions_labels = [list(labels_ids.keys())[list(labels_ids.values()).index(i)]]
          return predictions_labels[0]
    except Exception as e:
        traceback.print_exc()
        return f"An error has occured: {e}"
	# filepath = "train_data.csv"
# model_name = "gpt2"
# version = 1
# print(trainModel(filepath, model_name, version))