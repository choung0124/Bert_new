import os
import torch
import json
from torch.utils.data import DataLoader, TensorDataset, BatchSampler, SequentialSampler
from transformers import BertTokenizer, BertModel, BertPreTrainedModel
from tqdm import tqdm
from torch import nn
import warnings
import logging
import torch.nn as nn
logging.getLogger("transformers").setLevel(logging.ERROR)

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

batch_size = 8
num_epochs = 10
learning_rate = 5e-5

unique_ner_labels = set()
unique_relation_labels = set()
unique_ner_labels.add("O")

# Existing preprocessing functions
def preprocess_data(json_data, tokenizer, label_to_id, relation_to_id):
    ner_data = []
    re_data = []

    entities_dict = {entity["entityId"]: entity for entity in json_data["entities"]}

    relation_dict = {}
    for relation in json_data["relation_info"]:
        subject_id = relation["subjectID"].strip('"')  # Remove extra quotes
        obj_id = relation["objectId"].strip('"')  # Remove extra quotes
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

        # Process NER data
        entity_text = text[begin:end]
        entity_tokens = tokenizer.tokenize(entity_text)

        while current_idx < begin:
            ner_data.append((text[current_idx], "O"))
            current_idx += 1

        for i, token in enumerate(entity_tokens):
            if i == 0:
                label = f"B-{entity_type}-{entity_name}"
            else:
                label = f"I-{entity_type}-{entity_name}"

            # Add the label to the label_to_id dictionary if it's not present
            if label not in label_to_id:
                label_to_id[label] = len(label_to_id)

            ner_data.append((token, label))
            current_idx += 1

        current_idx = end

        # Add any new labels to the label_to_id mapping
        if f"{entity_type}-{entity_name}" not in label_to_id:
            label_to_id[f"{entity_type}-{entity_name}"] = len(label_to_id)

        # Process RE data
        if entity_id in relation_dict:
            for obj_id, rel_name in relation_dict[entity_id].items():
                if obj_id in entities_dict:
                    obj_entity = entities_dict[obj_id]
                    re_data.append({
                        'id': (entity_id, obj_id),
                        'subject': entity_text,
                        'object': text[obj_entity["span"]["begin"]:obj_entity["span"]["end"]],
                        'relation': rel_name,
                        'subject_tokens': entity_tokens,
                        'object_tokens': tokenizer.tokenize(text[obj_entity["span"]["begin"]:obj_entity["span"]["end"]])
                    })
                else:
                    #print(f"Warning: obj_id '{obj_id}' not found in entities_dict. Skipping this relation.")
                    continue

                # Add any new relations to the relation_to_id mapping
                if rel_name not in relation_to_id:
                    relation_to_id[rel_name] = len(relation_to_id)

    while current_idx < len(text):
        ner_data.append((text[current_idx], "O"))
        current_idx += 1
        
    if "O" not in label_to_id:
        label_to_id["O"] = len(label_to_id)

    return ner_data, re_data

json_directory = "test"
preprocessed_ner_data = []
preprocessed_re_data = []

label_to_id = {}
relation_to_id = {}

# Iterate through all JSON files in the directory
for file_name in os.listdir(json_directory):
    if file_name.endswith(".json"):
        json_path = os.path.join(json_directory, file_name)

        with open(json_path, "r") as json_file:
            json_data = json.load(json_file)

        ner_data, re_data = preprocess_data(json_data, tokenizer, label_to_id, relation_to_id)
        
        preprocessed_ner_data.append(ner_data)
        preprocessed_re_data.append(re_data)

        #print(f"Processed: {file_name}")
        #print(f"Number of entities: {len(json_data['entities'])}")
        #for entity in json_data['entities']:
        #    print(entity)


class BertForNERAndRE(BertPreTrainedModel):
    def __init__(self, config, num_ner_labels, num_re_labels):
        super().__init__(config)
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.ner_classifier = nn.Linear(config.hidden_size, num_ner_labels)
        self.re_classifier = nn.Linear(config.hidden_size, num_re_labels)
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
        re_logits = self.re_classifier(pooled_output)


        total_loss = 0
        if ner_labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            ner_loss = loss_fct(ner_logits.view(-1, self.config.num_ner_labels), ner_labels.view(-1))
            total_loss += ner_loss

        if re_labels is not None:
            re_logits_0 = torch.gather(re_logits, 1, re_indices[:, 0].unsqueeze(1)).squeeze(1)
            re_logits_1 = torch.gather(re_logits, 1, re_indices[:, 1].unsqueeze(1)).squeeze(1)
            re_logits = re_logits_0 + re_logits_1

            selected_re_labels_0 = torch.gather(re_labels, 1, re_indices[:, 0].unsqueeze(1)).squeeze(1)
            selected_re_labels_1 = torch.gather(re_labels, 1, re_indices[:, 1].unsqueeze(1)).squeeze(1)
            selected_re_labels = selected_re_labels_0 + selected_re_labels_1

            loss_fct = nn.CrossEntropyLoss()
            re_loss = loss_fct(re_logits, selected_re_labels)
            total_loss += re_loss

        output_dict = {
            "loss": total_loss if total_loss > 0 else None,
            "ner_logits": ner_logits,
            "re_logits": re_logits
        }

        return output_dict

# Preprocess and tokenize the NER and RE data
ner_input_ids, ner_attention_masks, ner_labels = [], [], []
re_input_ids, re_attention_masks, re_labels, re_indices_list = [], [], [], []

for ner_data, re_data in tqdm(zip(preprocessed_ner_data, preprocessed_re_data), desc="Tokenizing and aligning labels", total=len(preprocessed_ner_data)):
    # Tokenize and align NER labels
    ner_tokens, ner_labels_ = zip(*ner_data)
    encoded_ner = tokenizer.encode_plus(ner_tokens, is_split_into_words=True, add_special_tokens=True, padding="max_length", truncation=True, max_length=512, return_tensors="pt")

    ner_input_ids.append(encoded_ner["input_ids"])
    ner_attention_masks.append(encoded_ner["attention_mask"])

    aligned_ner_labels = [label_to_id[label] for label in ner_labels_]
    padded_ner_labels = aligned_ner_labels[:len(encoded_ner['input_ids'][0])] + [-100] * (len(encoded_ner['input_ids'][0]) - len(aligned_ner_labels))
    ner_labels.append(torch.LongTensor(padded_ner_labels))

    # Tokenize RE data
    for re_data_dict in re_data:
        subject_tokens = re_data_dict['subject_tokens']
        object_tokens = re_data_dict['object_tokens']

        subject_index = next((i for i, token in enumerate(ner_tokens) if ner_tokens[i:i+len(subject_tokens)] == subject_tokens), -1)
        object_index = next((i for i, token in enumerate(ner_tokens) if ner_tokens[i:i+len(object_tokens)] == object_tokens), -1)

        re_indices_list.append([subject_index, object_index])
        
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
        re_labels.append([relation_to_id[re_data_dict["relation"]]])
        

# Stack RE labels and pad the tensor
re_indices = torch.tensor(re_indices_list, dtype=torch.long).reshape(-1, 2)
re_labels = torch.stack([torch.tensor(labels) for labels in re_labels])
padding = torch.full((re_labels.shape[0], 512 - re_labels.shape[1]), -100, dtype=torch.long)
re_labels = torch.cat((re_labels, padding), dim=1)
print(f"re_labels shape: {re_labels.shape}")

# Convert lists to tensors
ner_input_ids = torch.cat(ner_input_ids)
ner_attention_masks = torch.cat(ner_attention_masks)
ner_labels = torch.stack(ner_labels)
re_input_ids = torch.cat(re_input_ids)
re_attention_masks = torch.cat(re_attention_masks)
re_labels = torch.stack(tuple(re_labels))

print(f"Shape of NER input ids: {ner_input_ids.shape}")
print(f"Shape of NER attention masks: {ner_attention_masks.shape}")
print(f"Shape of NER labels: {ner_labels.shape}")
print(f"Shape of RE input ids: {re_input_ids.shape}")
print(f"Shape of RE attention masks: {re_attention_masks.shape}")
print(f"Shape of RE labels: {re_labels.shape}")
print(f"Shape of RE indices: {re_indices.shape}")

assert ner_input_ids.shape == ner_attention_masks.shape == ner_labels.shape, "Mismatched shapes for NER input tensors"
assert re_input_ids.shape == re_attention_masks.shape == re_labels.shape, "Mismatched shapes for RE input tensors"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Defining re_loader if there is relation data
if len(re_input_ids) > 0 and len(re_data) > 0:
    re_dataset = TensorDataset(re_input_ids, re_attention_masks, re_labels)
    re_dataset_indices = list(range(len(re_input_ids)))
    re_sorted_indices = sorted(re_dataset_indices, key=lambda i: len(re_input_ids[i]))
    batch_size = 8
    re_batch_sampler = BatchSampler(SequentialSampler(re_sorted_indices), batch_size=batch_size, drop_last=False)
    re_loader = DataLoader(re_dataset, batch_sampler=re_batch_sampler)
    for batch in re_loader:
        input_ids, attention_masks, re_labels_batch = batch
        input_ids = input_ids.to(device)
        attention_masks = attention_masks.to(device)
        re_labels_batch = re_labels_batch.to(device)
        print(f"Shape of RE input ids batch: {input_ids.shape}")
        print(f"Shape of RE attention masks batch: {attention_masks.shape}")
        print(f"Shape of RE labels batch: {re_labels_batch.shape}")
        break
else:
    re_loader = None
    
# Create separate DataLoaders for NER and RE tasks
ner_dataset = TensorDataset(ner_input_ids, ner_attention_masks, ner_labels)
ner_dataset_indices = list(range(len(ner_input_ids)))
ner_sorted_indices = sorted(ner_dataset_indices, key=lambda i: len(ner_input_ids[i]))
batch_size = 8
ner_batch_sampler = BatchSampler(SequentialSampler(ner_sorted_indices), batch_size=batch_size, drop_last=False)
ner_dataloader = DataLoader(ner_dataset, batch_sampler=ner_batch_sampler)
for batch in ner_dataloader:
    input_ids, attention_masks, ner_labels_batch = batch
    input_ids = input_ids.to(device)
    attention_masks = attention_masks.to(device)
    ner_labels_batch = ner_labels_batch.to(device)
    print(f"Shape of NER input ids batch: {input_ids.shape}")
    print(f"Shape of NER attention masks batch: {attention_masks.shape}")
    print(f"Shape of NER labels batch: {ner_labels_batch.shape}")
    break

# Initialize the custom BERT model
model = BertForNERAndRE.from_pretrained("bert-base-uncased", num_ner_labels=len(label_to_id), num_re_labels=len(relation_to_id))

model.config.num_ner_labels = len(label_to_id)
model.config.num_re_labels = len(relation_to_id)

# Fine-tune the custom BERT model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
total_steps = len(ner_dataloader) * num_epochs  # You can adjust this based on your requirements

for epoch in tqdm(range(num_epochs), desc="Training epochs"):
    print(f'Epoch {epoch+1}/{num_epochs}')
    print('-' * 40)

    ner_epoch_loss = 0
    re_epoch_loss = 0
    ner_num_batches = 0
    re_num_batches = 0

    for ner_batch, re_batch in tqdm(zip(ner_dataloader, re_loader) if re_loader is not None else zip(ner_dataloader, [None] * len(ner_dataloader)), desc="Training batches"):
        # Training NER
        optimizer.zero_grad()
        input_ids, attention_masks, ner_labels = tuple(t.to(device) for t in ner_batch)
        re_indices = re_indices.to(device)  # Add this line to move re_indices to the correct device
        print(f"re_logits shape: {re_logits.shape}")
        print(f"re_indices shape: {re_indices.shape}")

        outputs = model(input_ids, attention_mask=attention_masks, ner_labels=ner_labels_batch, re_labels=re_labels_batch, re_indices=re_indices)
        ner_loss = outputs['loss']
        ner_epoch_loss += ner_loss.item()
        ner_num_batches += 1
        ner_loss.backward()
        optimizer.step()

        # Training RE
        if re_batch is not None:
            print(f"Length of re_batch: {len(re_batch)}")
            optimizer.zero_grad()
            input_ids, attention_masks, re_labels = tuple(t.to(device) for t in re_batch)
            re_indices = re_indices.to(device)  # Add this line to move re_indices to the correct device
            print(f"input_ids shape: {input_ids.shape}")
            print(f"re_labels shape: {re_labels.shape}")

            outputs = model(input_ids, attention_mask=attention_masks, re_labels=re_labels, re_indices=re_indices)
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

