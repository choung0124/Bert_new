import os
import torch
import json
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer, BertModel, BertPreTrainedModel, DistilBertPreTrainedModel, BertConfig, get_linear_schedule_with_warmup, AdamW, BertTokenizerFast, DistilBertConfig, DistilBertTokenizerFast, DistilBertModel
from tqdm import tqdm
from torch import nn
from torch.nn import CrossEntropyLoss
import warnings
import logging
import torch.nn as nn
import pickle
logging.getLogger("transformers").setLevel(logging.ERROR)
from torch.nn.utils.rnn import pad_sequence
from torch.cuda.amp import autocast, GradScaler
import traceback
import torch.nn.functional as F
import random

with open("preprocessed_ner_data.pkl", "rb") as f:
    preprocessed_ner_data = pickle.load(f)

with open("preprocessed_re_data.pkl", "rb") as f:
    preprocessed_re_data = pickle.load(f)
    
with open("label_to_id.pkl", "rb") as f:
    label_to_id = pickle.load(f)

with open("relation_to_id.pkl", "rb") as f:
    relation_to_id = pickle.load(f)
    
unique_ner_labels = set()
unique_relation_labels = set()

for item in preprocessed_ner_data:
    unique_ner_labels.add(item["subject_text"])
    unique_ner_labels.add(item["object_text"])

for item in preprocessed_re_data:
    unique_relation_labels.add(item["rel_name"])


# You can also save these mappings to pickle files if needed
with open("label_to_id.pkl", "wb") as f:
    pickle.dump(label_to_id, f)

with open("relation_to_id.pkl", "wb") as f:
    pickle.dump(relation_to_id, f)

#os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:64'
os.environ["TOKENIZERS_PARALLELISM"] = "false"
tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')  #device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu')
batch_size = 8
num_epochs = 4
learning_rate = 5e-5

class NERRE_Dataset(Dataset):
    def __init__(self, ner_data, re_data, tokenizer, max_length, label_to_id, relation_to_id):
        self.ner_data = ner_data
        self.re_data = re_data
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.label_to_id = label_to_id
        self.relation_to_id = relation_to_id
        self.ignore_label_index = max(self.label_to_id.values())  # Define ignore_label_index here

    def __len__(self):
        return len(self.ner_data)

    def __getitem__(self, idx):
        re_item = self.re_data[idx]
        tokens = re_item["sentence_tokens"]
        print(f"Item {idx} - Tokens: {tokens}")
        
        if not tokens:
        # You can either return a default value or skip this item
        # Here's an example of returning a default value:
            tokens = ["[UNK]"]

        inputs = self.tokenizer(tokens, padding='max_length', truncation=True, max_length=self.max_length, return_tensors='pt', return_offsets_mapping=True)
        #print(f"Tokenized output: {inputs}")
        input_ids = inputs['input_ids'].squeeze()
        attention_mask = inputs['attention_mask'].squeeze()
        offsets = inputs['offset_mapping'].squeeze()

        # Create ner_label_ids tensor with the same length as input_ids and initialize with a valid label index
        ner_label_ids = torch.full_like(input_ids, self.ignore_label_index, dtype=torch.long)
        #print(f"Item {idx} - Input_ids shape: {input_ids.shape}, Attention_mask shape: {attention_mask.shape}, NER_label_ids shape: {ner_label_ids.shape}")

        # Assign the indices of the subject and object tokens using the re_item values
        subject_start_idx = re_item['subject_start_idx']
        subject_end_idx = re_item['subject_end_idx']
        object_start_idx = re_item['object_start_idx']
        object_end_idx = re_item['object_end_idx']

        # Get the actual subject and object texts from the re_data
        subject_text = re_item['subject_text']
        object_text = re_item['object_text']

        # Assign appropriate labels for the subject and object tokens
        ner_label_ids[subject_start_idx:subject_end_idx+1] = self.label_to_id[subject_text]
        ner_label_ids[object_start_idx:object_end_idx+1] = self.label_to_id[object_text]

        re_labels = [self.relation_to_id[re_item['rel_name']]]

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'ner_labels': ner_label_ids,
            're_labels': torch.tensor(re_labels, dtype=torch.long),
            're_data': re_item
        }


def custom_collate_fn(batch, max_length):
    valid_batch = []

    for idx, item in enumerate(batch):
        try:
            if len(item['input_ids']) <= max_length:
                valid_batch.append(item)
        except KeyError:
            print(f"Skipping problematic data at index {idx} due to KeyError")
            continue
        except TypeError:
            print(f"Skipping problematic data at index {idx} due to TypeError")
            continue

    if not valid_batch:
        print("All items in the batch are problematic. Skipping the entire batch.")
        print(f"Problematic batch: {batch}")
        return None

    # Ensure the valid_batch has the same size as the original batch size
    while len(valid_batch) < len(batch):
        replacement_item = random.choice(batch)
        if len(replacement_item['input_ids']) <= max_length:
            valid_batch.append(replacement_item)

    try:
        input_ids = torch.stack([item['input_ids'] for item in valid_batch], dim=0)
        attention_mask = torch.stack([item['attention_mask'] for item in valid_batch], dim=0)
        ner_labels = torch.stack([item['ner_labels'] for item in valid_batch], dim=0)
        re_labels = torch.stack([item['re_labels'] for item in valid_batch], dim=0)
        re_data = [item['re_data'] for item in valid_batch]  # Add re_data to the collated batch
        print(f"Collated Batch - Padded Input_ids shape: {input_ids.shape}, Padded Attention_mask shape: {attention_mask.shape}, Padded NER_labels shape: {ner_labels.shape}")

    except RuntimeError:
        print("Skipping problematic data during padding")
        print("f"problematic batch: {batch}")
        return None

    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'ner_labels': ner_labels,
        're_labels': re_labels,
        're_data': re_data
    }





max_length = 128
if device.type == "cuda":
    num_workers = 2
else:
    num_workers = 6
dataset = NERRE_Dataset(preprocessed_ner_data, preprocessed_re_data, tokenizer, max_length, label_to_id, relation_to_id)
dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=lambda b: custom_collate_fn(b, max_length), num_workers=num_workers, shuffle=True)

# Print the first 5 batches from the DataLoader
for i, batch in enumerate(dataloader):
    if i < 5:
        print(f"Batch {i + 1}:")
        print(batch)
    else:
        break

# Check the length of the DataLoader
print(f"Number of batches in DataLoader: {len(dataloader)}")

class DistilBertForNERAndRE(DistilBertPreTrainedModel):
    def __init__(self, config, num_ner_labels, num_re_labels):
        super().__init__(config)

        self.num_ner_labels = num_ner_labels
        self.num_re_labels = num_re_labels

        self.distilbert = DistilBertModel(config)
        #self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.dropout = nn.Dropout(config.dropout)
        
        self.classifier = nn.Linear(config.hidden_size, self.num_ner_labels)

        # Define the bilinear layer for RE classification
        self.re_classifier = nn.Bilinear(config.hidden_size, config.hidden_size, self.num_re_labels)

        self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        head_mask=None,
        inputs_embeds=None,
        ner_labels=None,
        re_labels=None,
        re_data=None,
    ):
        #print("input_ids shape in forward:", input_ids.shape)
        #print("attention_mask shape in forward:", attention_mask.shape)
        outputs = self.distilbert(
            input_ids,
            attention_mask=attention_mask,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )

        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)
        ner_logits = self.classifier(sequence_output)

        if ner_labels is not None:
            
            loss_fct = nn.CrossEntropyLoss(ignore_index=dataset.ignore_label_index)
            #print("Attention mask shape:", attention_mask.shape)
            #print("NER logits shape:", ner_logits.shape)
            #print("NER labels shape:", ner_labels.shape)
            active_loss = attention_mask.view(-1).bool()
            active_logits = ner_logits.view(-1, self.num_ner_labels)[active_loss]
            active_labels = ner_labels.view(-1)[active_loss]

            # Check if there are any active logits and labels before calculating the loss
            if active_logits.shape[0] > 0 and active_labels.shape[0] > 0:
                ner_loss = loss_fct(active_logits, active_labels)
            else:
                ner_loss = None
        else:
            ner_loss = None

        if re_data is not None and len(re_data) > 0:
            #print(f"re_data: {re_data}")
            re_logits = []
            for b, batch_re_data in enumerate(re_data):
                batch_re_logits = []
                for rel in batch_re_data:
                    #print(f"rel: {rel}")
                    subject_start_idx = rel["subject_start_idx"]
                    subject_end_idx = rel["subject_end_idx"]
                    object_start_idx = rel["object_start_idx"]
                    object_end_idx = rel["object_end_idx"]

                    # Use the average of the hidden states for the subject and object tokens
                    subject_hidden_states = sequence_output[b, subject_start_idx:subject_end_idx+1].mean(dim=0)
                    object_hidden_states = sequence_output[b, object_start_idx:object_end_idx+1].mean(dim=0)

                    relation_logits = self.re_classifier(subject_hidden_states, object_hidden_states)
                    batch_re_logits.append(relation_logits.unsqueeze(0))

                batch_re_logits = torch.cat(batch_re_logits, dim=0)
                re_logits.append(batch_re_logits.unsqueeze(0))

            re_logits = torch.cat(re_logits, dim=0)
        else:
            re_logits = None

        return {'ner_logits': ner_logits, 're_logits': re_logits, 'ner_loss': ner_loss}


# Set up the configuration, model, and tokenizer
config = DistilBertConfig.from_pretrained("distilbert-base-uncased")
tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")

# Initialize the model with the given configuration
num_ner_labels = len(label_to_id)
num_re_labels = len(relation_to_id)
model = DistilBertForNERAndRE(config, num_ner_labels, num_re_labels)
model = model.to(device)

# Mixed precision training
#scaler = torch.cuda.amp.GradScaler()

# Prepare the optimizer and learning rate scheduler
optimizer = AdamW(model.parameters(), lr=learning_rate)
num_training_steps = len(dataloader) * num_epochs
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=num_training_steps)
accumulation_steps = 1  # Increase this value based on the desired accumulation steps.
accumulation_counter = 0

# Define separate loss functions for NER and RE tasks
ner_loss_fn = CrossEntropyLoss(ignore_index=-100)
re_loss_fn = CrossEntropyLoss(ignore_index=-1)

# Initialize the scaler if device is CUDA
if device.type == "cuda":
    scaler = torch.cuda.amp.GradScaler()
else:
    scaler = None

for epoch in range(num_epochs):
    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{num_epochs}")

    for step, batch in enumerate(progress_bar):
        
        try:
            model.train()

            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            ner_labels = batch['ner_labels'].to(device)
            re_labels = batch['re_labels'].to(device)
            re_data = batch['re_data']

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                ner_labels=ner_labels,
                re_labels=re_labels if len(re_data) > 0 else None,  # Add this condition
                re_data=re_data,
            )

            # Get NER and RE logits from model output
            ner_logits = outputs["ner_logits"]
            re_logits = outputs["re_logits"]

            # Calculate NER loss
            ner_loss = outputs["ner_loss"]

            # Calculate RE loss
            re_loss = 0
            if re_logits is not None:  # Add this condition
                for b, batch_re_labels in enumerate(re_labels):
                    re_loss += re_loss_fn(re_logits[b].view(-1, re_logits.size(-1)), batch_re_labels.view(-1))

                # Normalize RE loss by the number of samples
                re_loss /= re_logits.size(0)

            # Combine NER and RE losses using a weighted sum
            loss_weight = 0.5  # Adjust this value based on the importance of each task
            total_loss = loss_weight * ner_loss + (1 - loss_weight) * re_loss

            # Update counter
            accumulation_counter += 1

            # Perform optimization step and zero gradients if counter has reached accumulation steps
            if accumulation_counter % accumulation_steps == 0:
                if scaler is not None:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            # Update progress bar
            progress_bar.set_postfix({"NER Loss": ner_loss.item(), "RE Loss": re_loss.item(), "Total Loss": total_loss.item()})

        except Exception as e:
            print(f"Skipping batch due to error: {e}")
            continue


# Save the fine-tuned custom BERT model and tokenizer
output_dir = "models/combined"
os.makedirs(output_dir, exist_ok=True)
model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)

# Save the label_to_id and relation_to_id mappings
with open(os.path.join(output_dir, "label_to_id.json"), "w") as f:
    json.dump(label_to_id, f)

with open(os.path.join(output_dir, "relation_to_id.json"), "w") as f:
    json.dump(relation_to_id, f)
