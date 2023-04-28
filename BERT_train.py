import os
import torch
import json
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer, BertModel, BertPreTrainedModel, BertConfig, get_linear_schedule_with_warmup, AdamW
from tqdm import tqdm
from torch import nn
import warnings
import logging
import torch.nn as nn
import itertools
logging.getLogger("transformers").setLevel(logging.ERROR)
from torch.nn.utils.rnn import pad_sequence

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

device = torch.device("cpu")
batch_size = 8
num_epochs = 4
learning_rate = 5e-5

unique_ner_labels = set()
unique_relation_labels = set()
unique_ner_labels.add("O")

# Existing preprocessing functions
import itertools

def preprocess_data(json_data, tokenizer, label_to_id, relation_to_id):
    ner_data = []
    re_data = []
    re_indices = []

    entities_dict = {entity["entityId"]: entity for entity in json_data["entities"]}
    entity_ids = set(entities_dict.keys())

    relation_dict = {}
    for relation in json_data["relation_info"]:
        subject_id = relation["subjectID"].strip('"')
        obj_id = relation["objectId"].strip('"')
        if subject_id not in relation_dict:
            relation_dict[subject_id] = {}
        relation_dict[subject_id][obj_id] = relation["rel_name"]

    text = json_data["text"]
    current_idx = 0
    for entity in sorted(json_data["entities"], key=lambda x: x["span"]["begin"]):
        begin, end = entity["span"]["begin"], entity["span"]["end"]
        entity_type = entity["entityType"]
        entity_id = entity["entityId"].strip('"')
        entity_name = entity["entityName"]

        entity_text = text[begin:end]
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
            current_idx += 1

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

    preprocessed_data = []
    re_labels = [relation_to_id[relation['relation']] for relation in re_data]
    
    # Check if there are relations before appending the data
    if len(re_data) > 0:
        preprocessed_data.append({
            'ner_data': ner_data,
            're_data': re_data,
            're_indices': re_indices,
            're_labels': re_labels  # Add the re_labels list here
        })

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
        #print("index:", idx, "item:", item)
        # If re_labels is empty, return None
        if len(item['re_labels']) == 0:
            return None
        # Tokenize the text and prepare inputs
        tokens = [token for token, _, _ in item['ner_data']]
        ner_labels = [label for _, label, _ in item['ner_data']]
        ner_label_ids = [self.label_to_id[label] for label in ner_labels]

        inputs = self.tokenizer(tokens, padding='max_length', truncation=True, max_length=512, return_tensors='pt')

        input_ids = inputs['input_ids'].squeeze()
        attention_mask = inputs['attention_mask'].squeeze()
        token_type_ids = inputs['token_type_ids'].squeeze()

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
            'token_type_ids': token_type_ids,
            'ner_labels': torch.tensor(ner_label_ids, dtype=torch.long),
            're_labels': torch.tensor(item['re_labels'], dtype=torch.long),
            're_indices': torch.tensor(list(zip(subject_indices, object_indices)), dtype=torch.long) if subject_indices is not None and object_indices is not None else None
        }

def pad_relation_data(data, max_relations, padding_value=-1):
    padded_data = data + [torch.tensor([padding_value, padding_value], dtype=torch.long)] * (max_relations - len(data))
    return padded_data

def custom_collate_fn(batch):
    # Remove None values from the batch
    batch = [item for item in batch if item is not None]

    # Pad input_ids, attention_mask, and token_type_ids
    input_ids = pad_sequence([item['input_ids'] for item in batch], batch_first=True)
    attention_mask = pad_sequence([item['attention_mask'] for item in batch], batch_first=True)
    token_type_ids = pad_sequence([item['token_type_ids'] for item in batch], batch_first=True)

    # Pad ner_labels
    ner_labels = pad_sequence([item['ner_labels'] for item in batch], batch_first=True, padding_value=-100)

    # Find the maximum number of relations in the batch
    max_relations = max([len(item['re_labels']) for item in batch])

    # Pad re_labels
    padded_re_labels = [pad_relation_data(item['re_labels'], max_relations) for item in batch]
    re_indices = torch.stack(padded_re_indices)
    
    # Pad re_indices
    re_indices = [item['re_indices'] for item in batch if 're_indices' in item and item['re_indices'] is not None]

    if not re_indices:
        re_indices = None
    else:
        # Find the maximum number of relations for re_indices in the batch
        max_re_indices_relations = max([len(item) for item in re_indices])

        # Pad re_indices
        padded_re_indices = [pad_relation_data(item, max_re_indices_relations, padding_value=(-1, -1)) for item in re_indices]
        re_indices = torch.tensor(padded_re_indices, dtype=torch.long)
        
        
    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'token_type_ids': token_type_ids,
        'ner_labels': ner_labels,
        're_labels': re_labels,
        're_indices': re_indices  # Add the 're_indices' key here
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

max_length = 128
if device.type == "cuda":
    num_workers = 0
else:
    num_workers = 12
dataset = NERRE_Dataset(preprocessed_data, tokenizer, max_length, label_to_id, relation_to_id)
dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=custom_collate_fn, num_workers=num_workers)

class BertForNERAndRE(BertPreTrainedModel):
    def __init__(self, config, num_ner_labels, num_re_labels):
        super().__init__(config)
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.ner_classifier = nn.Linear(config.hidden_size, num_ner_labels)
        self.re_classifier = nn.Bilinear(config.hidden_size, config.hidden_size, num_re_labels)
        self.config.num_ner_labels = num_ner_labels
        self.config.num_re_labels = num_re_labels

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        ner_labels=None,
        re_labels=None,
        re_indices=None,
    ):
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )

        sequence_output, pooled_output = outputs.last_hidden_state, outputs.pooler_output

        sequence_output = self.dropout(sequence_output)
        pooled_output = self.dropout(pooled_output)

        ner_logits = self.ner_classifier(sequence_output)

        # Extract subject and object hidden states from sequence_output using re_indices
        subject_hidden_states = sequence_output[torch.arange(sequence_output.size(0)).unsqueeze(1), re_indices[:, 0]]
        object_hidden_states = sequence_output[torch.arange(sequence_output.size(0)).unsqueeze(1), re_indices[:, 1]]

        # Compute RE logits using the bilinear layer and the extracted hidden states
        re_logits = self.re_classifier(subject_hidden_states, object_hidden_states)

        total_loss = 0
        if ner_labels is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=-1)
            ner_loss = loss_fct(ner_logits.view(-1, self.config.num_ner_labels), ner_labels.view(-1))
            total_loss += ner_loss


        output_dict = {
            "loss": total_loss if total_loss > 0 else None,
            "ner_logits": ner_logits,
            "re_logits": re_logits
        }

        return output_dict


# Initialize the custom BERT model
from transformers import BertTokenizer, BertConfig, AdamW, get_linear_schedule_with_warmup

# Set up the configuration, model, and tokenizer
config = BertConfig.from_pretrained("bert-base-uncased")
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# Initialize the model with the given configuration
num_ner_labels = len(label_to_id)
num_re_labels = len(relation_to_id)
model = BertForNERAndRE(config, num_ner_labels, num_re_labels)
model = model.to(device)
#if torch.cuda.device_count() > 1:
    #model = nn.DataParallel(model)

# Prepare the optimizer and learning rate scheduler
optimizer = AdamW(model.parameters(), lr=3e-5)
num_training_steps = len(dataloader) * num_epochs
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=num_training_steps)

for epoch in range(num_epochs):
    
    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{num_epochs}")
    
    for batch in dataloader:
        # Skip the iteration if the batch is empty
        if not batch:
            continue
            
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        token_type_ids = batch['token_type_ids'].to(device)
        ner_labels = batch['ner_labels'].to(device)
        re_labels = batch['re_labels'].to(device)
        if batch['re_indices'] is not None:
            re_indices = batch['re_indices'].to(device) 
        else:
            re_indices = None

        try:
            # Forward pass
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                ner_labels=ner_labels,
                re_labels=re_labels,
                re_indices=re_indices,
            )

            loss = outputs["loss"]
            loss.backward()

            # Update parameters and the learning rate
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            
            progress_bar.set_postfix({"loss": loss.item()})

        except Exception as e:
            print(f"Skipping batch due to error: {e}")
            continue
    progress_bar.close()


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

