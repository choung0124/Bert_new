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
import itertools
logging.getLogger("transformers").setLevel(logging.ERROR)
from torch.nn.utils.rnn import pad_sequence
from torch.cuda.amp import autocast, GradScaler

os.environ["TOKENIZERS_PARALLELISM"] = "false"
tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
batch_size = 1
num_epochs = 4
learning_rate = 5e-5

unique_ner_labels = set()
unique_relation_labels = set()
unique_ner_labels.add("O")

# Existing preprocessing functions
def preprocess_data(json_data, tokenizer, label_to_id, relation_to_id):
    ner_data = []
    re_data = []
    re_indices = []

    entities_dict = {entity["entityId"]: entity for entity in json_data["entities"]}
    entity_ids = set(entities_dict.keys())

    relation_dict = {}
    for relation in json_data["relation_info"]:
        subject_id = relation["subjectID"]
        obj_id = relation["objectId"]
        if subject_id not in relation_dict:
            relation_dict[subject_id] = {}
        relation_dict[subject_id][obj_id] = relation["rel_name"]

    text = json_data["text"]
    current_idx = 0
    for entity in sorted(json_data["entities"], key=lambda x: x["span"]["begin"]):
        begin, end = entity["span"]["begin"], entity["span"]["end"]
        entity_type = entity["entityType"]
        entity_id = entity["entityId"]
        entity_name = entity["entityName"]

        entity_text = text[begin:end].strip()
        entity_tokens = tokenizer.tokenize(entity_text)

        while current_idx < begin:
            ner_data.append((text[current_idx], "O", len(ner_data)))
            current_idx += 1

        for i, token in enumerate(entity_tokens):
            if i == 0:
                label = f"B-{entity_type}-{entity_name}"
            else:
                label = f"I-{entity_type}-{entity_name}"

            if label not in label_to_id:
                label_to_id[label] = len(label_to_id)

            ner_data.append((token, label, len(ner_data)))
            current_idx += len(token)

        current_idx = end

        if f"{entity_type}-{entity_name}" not in label_to_id:
            label_to_id[f"{entity_type}-{entity_name}"] = len(label_to_id)

    for entity_id_1, entity_id_2 in itertools.combinations(entity_ids, 2):
        if entity_id_1 in relation_dict and entity_id_2 in relation_dict[entity_id_1]:
            rel_name = relation_dict[entity_id_1][entity_id_2]
            entity_1 = entities_dict[entity_id_1]
            entity_2 = entities_dict[entity_id_2]
            re_data.append({
                'id': (entity_id_1, entity_id_2),
                'subject': text[entity_1["span"]["begin"]:entity_1["span"]["end"]],
                'object': text[entity_2["span"]["begin"]:entity_2["span"]["end"]],
                'relation': rel_name,
                'subject_tokens': tokenizer.tokenize(text[entity_1["span"]["begin"]:entity_1["span"]["end"]]),
                'object_tokens': tokenizer.tokenize(text[entity_2["span"]["begin"]:entity_2["span"]["end"]])
            })

            if rel_name not in relation_to_id:
                relation_to_id[rel_name] = len(relation_to_id)

            subject_start_idx = [idx for token, label, idx in ner_data if label == f"B-{entity_1['entityType']}-{entity_1['entityName']}"][0]
            object_start_idx = [idx for token, label, idx in ner_data if label == f"B-{entity_2['entityType']}-{entity_2['entityName']}"][0]
            re_indices.append((subject_start_idx, object_start_idx))

    while current_idx < len(text):
        ner_data.append((text[current_idx], "O", len(ner_data)))
        current_idx += 1

    if "O" not in label_to_id:
        label_to_id["O"] = len(label_to_id)

    re_labels = [relation_to_id[relation['relation']] for relation in re_data]

    preprocessed_data = [{
        'ner_data': ner_data,
        're_data': re_data,
        're_indices': re_indices,
        're_labels': re_labels
    }] if len(re_data) > 0 else []

    return preprocessed_data


class NERRE_Dataset(Dataset):
    def __init__(self, data, tokenizer, max_length, label_to_id, relation_to_id):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.label_to_id = label_to_id
        self.relation_to_id = relation_to_id

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        
        # If re_labels is empty, return None
        if len(item['re_labels']) == 0:
            return None
        
        # Tokenize the text and prepare inputs
        tokens = [token for token, _, _ in item['ner_data']]
        ner_labels = [label for _, label, _ in item['ner_data']]
        ner_label_ids = [self.label_to_id[label] for label in ner_labels]

        inputs = self.tokenizer(tokens, padding='max_length', truncation=True, max_length=64, return_tensors='pt')

        input_ids = inputs['input_ids'].squeeze()
        attention_mask = inputs['attention_mask'].squeeze()
        #token_type_ids = inputs['token_type_ids'].squeeze()

        if len(item['re_indices']) == 0:
            # Handle the case when re_indices is empty
            # Return None for subject_indices and object_indices
            subject_indices = None
            object_indices = None
        else:
            # Split the re_indices tuples into separate lists of subject and object indices
            subject_indices, object_indices = zip(*item['re_indices'])

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            #'token_type_ids': token_type_ids,
            'ner_labels': torch.tensor(ner_label_ids, dtype=torch.long),
            're_labels': torch.tensor(item['re_labels'], dtype=torch.long),
            're_indices': torch.tensor(list(zip(subject_indices, object_indices)), dtype=torch.long) if subject_indices is not None and object_indices is not None else None
        }

def pad_relation_indices(re_indices_list, max_relations, padding_value=-1):
    padded_re_indices = []
    for re_indices in re_indices_list:
        padding_tensor = torch.full((max_relations - len(re_indices), 2), padding_value, dtype=torch.long)
        padded_indices = torch.cat([re_indices, padding_tensor], dim=0)
        padded_re_indices.append(padded_indices)
    return torch.stack(padded_re_indices, dim=0)


def custom_collate_fn(batch):
    # Remove None values from the batch
    #batch = [item for item in batch if item is not None]

    # Pad input_ids, attention_mask, and token_type_ids
    input_ids = pad_sequence([item['input_ids'] for item in batch], batch_first=True)
    attention_mask = pad_sequence([item['attention_mask'] for item in batch], batch_first=True)
    #token_type_ids = pad_sequence([item['token_type_ids'] for item in batch], batch_first=True)

    # Pad ner_labels
    ner_labels = pad_sequence([item['ner_labels'] for item in batch], batch_first=True, padding_value=-100)
    
    # Pad re_labels
    re_labels = pad_sequence([item['re_labels'] for item in batch], batch_first=True, padding_value=-1)

    # Pad re_indices
    re_indices_list = [item['re_indices'] for item in batch if item['re_indices'] is not None]

    if len(re_indices_list) > 0:
        max_relations = max(len(re_indices) for re_indices in re_indices_list)
        re_indices = pad_relation_indices(re_indices_list, max_relations)
        re_indices = torch.stack(tuple(re_indices), dim=0)  # concatenate tensors along a new dimension
    else:
        re_indices = None

    # Return the final dictionary
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        #"token_type_ids": token_type_ids,
        "ner_labels": ner_labels,
        "re_labels": re_labels,
        "re_indices": re_indices,
    }

label_to_id = {}
relation_to_id = {}

json_directory = "test"
preprocessed_data = []

# Iterate through all JSON files in the directory
def validate_json(json_data):
    # Check if the necessary keys are present
    if "entities" not in json_data or "relation_info" not in json_data or "text" not in json_data:
        return False

    # Check if there are entities and relations
    if len(json_data["entities"]) == 0 or len(json_data["relation_info"]) == 0:
        return False

    # Additional validation criteria can be added here based on your data format

    return True

for file_name in os.listdir(json_directory):
    if file_name.endswith(".json"):
        json_path = os.path.join(json_directory, file_name)

        try:
            with open(json_path, "r") as json_file:
                json_data = json.load(json_file)

            if validate_json(json_data):
                preprocessed_file_data = preprocess_data(json_data, tokenizer, label_to_id, relation_to_id)
                preprocessed_data.extend(preprocessed_file_data)
            else:
                print(f"Skipping {json_path} due to invalid JSON data")
        except json.JSONDecodeError as e:
            print(f"Error loading {json_path}: {e}")
            continue

max_length = 64
if device.type == "cuda":
    num_workers = 8
else:
    num_workers = 8
dataset = NERRE_Dataset(preprocessed_data, tokenizer, max_length, label_to_id, relation_to_id)
dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=custom_collate_fn, num_workers=num_workers, shuffle=True)

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
        #token_type_ids=None,
        #position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        ner_labels=None,
        re_labels=None,
        re_indices=None,
    ):
        outputs = self.distilbert(
            input_ids,
            attention_mask=attention_mask,
            #token_type_ids=token_type_ids,
            #position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )

        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)
        ner_logits = self.classifier(sequence_output)

        if ner_labels is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=-1)
            active_loss = attention_mask.view(-1) == 1
            active_logits = ner_logits.view(-1, self.num_ner_labels)[active_loss]  # Use self.num_ner_labels instead of self.num_labels
            active_labels = ner_labels.view(-1)[active_loss]
            ner_loss = loss_fct(active_logits, active_labels)
        else:
            ner_loss = None

        if re_indices is not None and re_indices.size(1) > 0:
            # Initialize an empty tensor for storing RE logits
            re_logits = torch.zeros(sequence_output.size(0), sequence_output.size(1), re_indices.size(1), self.num_re_labels).to(sequence_output.device)

            # Iterate over the samples and sentences in the batch
            for b in range(sequence_output.size(0)):
                for s in range(sequence_output.size(1)):
                    for i in range(re_indices.size(1)):
                        if re_indices[b, i, 0] != -1 and re_indices[b, i, 1] != -1:
                            # Extract subject and object hidden states from sequence_output using re_indices
                            subject_hidden_states = sequence_output[b, s, re_indices[b, i, 0]]
                            object_hidden_states = sequence_output[b, s, re_indices[b, i, 1]]

                            # Compute RE logits using the bilinear layer and the extracted hidden states
                            relation_logits = self.re_classifier(subject_hidden_states, object_hidden_states)
                            re_logits[b, s, i, :] = relation_logits
        else:
            re_logits = None  # Set re_logits to None if re_indices is None or empty

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
scaler = torch.cuda.amp.GradScaler()

# Prepare the optimizer and learning rate scheduler
optimizer = AdamW(model.parameters(), lr=learning_rate)
num_training_steps = len(dataloader) * num_epochs
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=num_training_steps)
accumulation_steps = 32  # Increase this value based on the desired accumulation steps.
accumulation_counter = 0

# Define separate loss functions for NER and RE tasks
ner_loss_fn = CrossEntropyLoss(ignore_index=-100)
re_loss_fn = CrossEntropyLoss(ignore_index=-1)

for epoch in range(num_epochs):
    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{num_epochs}")

    for step, batch in enumerate(progress_bar):
        try:
            model.train()

            input_ids = batch['input_ids'].view(-1, batch['input_ids'].size(-1)).to(device)
            attention_mask = batch['attention_mask'].view(-1, batch['attention_mask'].size(-1)).to(device)
            #token_type_ids = batch['token_type_ids'].view(-1, batch['token_type_ids'].size(-1)).to(device)
            ner_labels = batch['ner_labels'].view(-1).to(device)
            re_labels = batch['re_labels'].to(device)
            re_indices = batch['re_indices']

            if re_indices is not None:
                re_indices = re_indices.to(device)
            else:
                re_indices = None

            with autocast():
                # Forward pass
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    #token_type_ids=token_type_ids,
                    ner_labels=ner_labels,
                    re_labels=re_labels,
                    re_indices=re_indices if re_indices is not None else None,
                )

                # Get NER and RE logits from model output
                ner_logits = outputs["ner_logits"]
                re_logits = outputs["re_logits"]

                # Calculate NER loss
                ner_loss = ner_loss_fn(ner_logits.view(-1, ner_logits.size(-1)), ner_labels.view(-1))

                # Calculate RE loss
                re_loss = 0
                for b in range(re_logits.size(0)):
                    for s in range(re_logits.size(1)):
                        re_loss += re_loss_fn(re_logits[b, s].view(-1, re_logits.size(-1)), re_labels[b, s].view(-1))

                # Normalize RE loss by the number of sentences
                re_loss /= (re_logits.size(0) * re_logits.size(1))

                # Combine NER and RE losses using a weighted sum
                loss_weight = 0.5  # Adjust this value based on the importance of each task
                total_loss = loss_weight * ner_loss + (1 - loss_weight) * re_loss

            scaler.scale(total_loss).backward()

            # Update counter
            accumulation_counter += 1

            # Perform optimization step and zero gradients if counter has reached accumulation steps
            if accumulation_counter % accumulation_steps == 0:
                scaler.step(optimizer)
                scaler.update()
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
