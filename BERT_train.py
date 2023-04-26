import os
import torch
import json
from torch.utils.data import DataLoader, TensorDataset
from transformers import BertTokenizer, BertModel, BertPreTrainedModel
from tqdm import tqdm
from torch import nn
import warnings
import logging
logging.getLogger("transformers").setLevel(logging.ERROR)

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

batch_size = 8
num_epochs = 4
learning_rate = 5e-5

unique_ner_labels = set()
unique_relation_labels = set()
unique_ner_labels.add("O")

# Existing preprocessing functions
def preprocess_data(json_data, tokenizer):
    ner_data = []
    re_data = []

    entities_dict = {entity["entityId"]: entity for entity in json_data["entities"]}

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

        # Process NER data
        entity_text = text[begin:end]
        entity_tokens = tokenizer.tokenize(entity_text)

        while current_idx < begin:
            ner_data.append((text[current_idx], "O"))
            current_idx += 1

        for i, token in enumerate(entity_tokens):
            ner_data.append((token, f"{entity_type}" if i == 0 else f"I-{entity_type}"))
            current_idx += 1

        current_idx = end

        # Process RE data
        if entity_id in relation_dict:
            for obj_id, rel_name in relation_dict[entity_id].items():
                obj_entity = entities_dict[obj_id]
                re_data.append({
                    'id': (entity_id, obj_id),
                    'subject': entity_text,
                    'object': text[obj_entity["span"]["begin"]:obj_entity["span"]["end"]],
                    'relation': rel_name,
                    'subject_tokens': entity_tokens,
                    'object_tokens': tokenizer.tokenize(text[obj_entity["span"]["begin"]:obj_entity["span"]["end"]])
                })

    while current_idx < len(text):
        ner_data.append((text[current_idx], "O"))
        current_idx += 1

    return ner_data, re_data

json_directory = "test"
preprocessed_ner_data = []
preprocessed_re_data = []

# Iterate through all JSON files in the directory
for file_name in os.listdir(json_directory):
    if file_name.endswith(".json"):
        json_path = os.path.join(json_directory, file_name)

        # Load the JSON data
        with open(json_path, "r") as json_file:
            json_data = json.load(json_file)

        # Preprocess the data for NER tasks
        ner_data, relation_dict = preprocess_data(json_data, tokenizer)
        preprocessed_ner_data.append(ner_data)
        
        # Preprocess the data for RE tasks
        re_data = preprocess_data(json_data, tokenizer)
        #print(re_data)# <-- Call preprocess_re
        preprocessed_re_data.append(re_data)  # <-- Store the processed RE data

        #print(f"Processed: {file_name}")
        #print(f"Number of entities: {len(json_data['entities'])}")
        #for entity in json_data['entities']:
        #    print(entity)

            
label_to_id = {label: idx for idx, label in enumerate(sorted(unique_ner_labels))}
relation_to_id = {relation: idx for idx, relation in enumerate(sorted(unique_relation_labels))}

# New custom model based on BERT
class BertForNERAndRE(BertPreTrainedModel):
    def __init__(self, config, num_ner_labels, num_re_labels):
        super().__init__(config)
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.ner_classifier = nn.Linear(config.hidden_size, num_ner_labels)
        self.re_classifier = nn.Linear(config.hidden_size, num_re_labels)

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
    ):
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )

        sequence_output = outputs[0]
        pooled_output = outputs[1]

        sequence_output = self.dropout(sequence_output)
        pooled_output = self.dropout(pooled_output)

        ner_logits = self.ner_classifier(sequence_output)
        re_logits = self.re_classifier(pooled_output)

        total_loss = 0
        if ner_labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            ner_loss = loss_fct(ner_logits.view(-1, self.config.num_ner_labels), ner_labels.view(-1))
            total_loss += ner_loss

        if re_labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            re_loss = loss_fct(re_logits.view(-1, self.config.num_re_labels), re_labels.view(-1))
            total_loss += re_loss

        return (total_loss, ner_logits, re_logits) if total_loss > 0 else (ner_logits, re_logits)

# Preprocess and tokenize the NER and RE data
ner_input_ids, ner_attention_masks, ner_labels = [], [], []
re_input_ids, re_attention_masks, re_labels = [], [], []

# Tokenize NER and RE data more efficiently
ner_input_ids, ner_attention_masks, ner_labels = [], [], []
re_input_ids, re_attention_masks, re_labels = [], [], []

for ner_data, re_data in tqdm(zip(preprocessed_ner_data, preprocessed_re_data), desc="Tokenizing and aligning labels"):
    # Tokenize and align NER labels
    ner_tokens, ner_labels_ = zip(*ner_data)
    encoded_ner = tokenizer.encode_plus(ner_tokens, is_split_into_words=True, add_special_tokens=True, padding="max_length", truncation=True, max_length=512, return_tensors="pt")
    ner_input_ids.append(encoded_ner["input_ids"])
    ner_attention_masks.append(encoded_ner["attention_mask"])

    aligned_ner_labels = [label_to_id[label] for label in ner_labels_]
    padded_ner_labels = aligned_ner_labels[:512] + [-100] * (512 - len(aligned_ner_labels))
    ner_labels.append(torch.tensor(padded_ner_labels))

    # Tokenize RE data
    for re_data_dict in re_data:
        encoded_re = tokenizer.encode_plus(
            re_data_dict["subject_tokens"],
            re_data_dict["object_tokens"],
            add_special_tokens=True,
            padding="max_length",
            truncation=True,
            max_length=512,
            return_tensors="pt"
        )

        re_input_ids.append(encoded_re["input_ids"])
        re_attention_masks.append(encoded_re["attention_mask"])
        re_labels.append(relation_label_to_id[re_data_dict["relation"]])

# Convert lists to tensors
ner_input_ids = torch.cat(ner_input_ids)
ner_attention_masks = torch.cat(ner_attention_masks)
ner_labels = torch.cat(ner_labels)
re_input_ids = torch.cat(re_input_ids)
re_attention_masks = torch.cat(re_attention_masks)
re_labels = torch.tensor(re_labels)

# Defining re_loader if there is relation data
if len(re_input_ids) > 0:
    re_dataset = TensorDataset(re_input_ids, re_attention_masks, re_labels)
    re_loader = DataLoader(re_dataset, batch_size=batch_size)
else:
    re_loader = None

# Create separate DataLoaders for NER and RE tasks
ner_dataset = TensorDataset(ner_input_ids, ner_attention_masks, ner_labels)
ner_loader = DataLoader(ner_dataset, batch_size=batch_size)


# Initialize the custom BERT model
model = BertForNERAndRE.from_pretrained("bert-base-uncased", num_ner_labels=len(label_to_id), num_re_labels=len(relation_to_id))

model.config.num_ner_labels = len(label_to_id)
model.config.num_re_labels = len(relation_to_id)

# Fine-tune the custom BERT model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
total_steps = len(ner_loader) * num_epochs  # You can adjust this based on your requirements

for epoch in tqdm(range(num_epochs), desc="Training epochs"):
    print(f'Epoch {epoch+1}/{num_epochs}')
    print('-' * 40)

    ner_epoch_loss = 0
    re_epoch_loss = 0
    ner_num_batches = 0
    re_num_batches = 0

    for ner_batch, re_batch in tqdm(zip(ner_loader, re_loader) if re_loader is not None else zip(ner_loader, [None] * len(ner_loader)), desc="Training batches"):
        # Training NER
        optimizer.zero_grad()
        input_ids, attention_masks, ner_labels = tuple(t.to(device) for t in ner_batch)
        outputs = model(input_ids, attention_mask=attention_masks, ner_labels=ner_labels)
        ner_loss = outputs[0]
        ner_epoch_loss += ner_loss.item()
        ner_num_batches += 1
        ner_loss.backward()
        optimizer.step()

        # Training RE
        if re_batch is not None:
            optimizer.zero_grad()
            input_ids, attention_masks, re_labels = tuple(t.to(device) for t in re_batch)
            outputs = model(input_ids, attention_mask=attention_masks, re_labels=re_labels)
            re_loss = outputs[0]
            re_epoch_loss += re_loss.item()
            re_num_batches += 1
            re_loss.backward()
            optimizer.step()

    ner_epoch_loss /= ner_num_batches
    re_epoch_loss /= re_num_batches if re_num_batches > 0 else 1

    print(f'Train loss NER: {ner_epoch_loss} Train loss RE: {re_epoch_loss}')


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

