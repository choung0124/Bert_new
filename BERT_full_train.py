import os
import torch
import json
from torch.utils.data import DataLoader, TensorDataset
from transformers import BertTokenizer, BertModel, BertPreTrainedModel
from tqdm import tqdm
from torch import nn

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

batch_size = 8
num_epochs = 4
learning_rate = 5e-5

unique_ner_labels = set()
unique_relation_labels = set()
unique_ner_labels.add("O")


# Existing preprocessing functions
def preprocess_re(json_data, relation_dict, tokenizer):
    re_data = []
    
    for subject_id, obj_dict in relation_dict.items():
        if subject_id not in json_data["entities"]:
            continue
        subject = json_data["entities"][subject_id]["entityName"]
        for obj_id, relation_name in obj_dict.items():
            if obj_id not in json_data["entities"]:
                continue
            obj = json_data["entities"][obj_id]["entityName"]
            unique_relation_labels.add(relation_name)  # <-- Add this line
            re_data.append((subject_id, obj_id, relation_name, obj))
            print(f"Processed relation: ({subject}, {obj}, {relation_name})")
            
    return re_data

def preprocess_ner(json_data, tokenizer):
    ner_data = []
    relation_dict = {}
    
    # Extract the relation info and build a dictionary
    for relation in json_data["relation_info"]:
        subject_id, obj_id = relation["subjectID"], relation["objectId"]
        if subject_id not in relation_dict:
            relation_dict[subject_id] = {}
        relation_dict[subject_id][obj_id] = relation["rel_name"]
        
    for entity in json_data["entities"]:
        begin = entity["span"]["begin"]
        end = entity["span"]["end"]
        entity_type = entity["entityType"]
        entity_text = json_data["text"][begin:end]
        relation_ids = []

        # Check if the entity has any relations
        if entity["entityId"] in relation_dict:
            for obj_id, rel_name in relation_dict[entity["entityId"]].items():
                relation_ids.append((obj_id, rel_name))

        ner_data.append((begin, end, entity_type, entity_text, relation_ids))
    
    ner_data.sort(key=lambda x: x[0])
    
    text = json_data["text"]
    ner_tags = []
    current_idx = 0
    
    for begin, end, entity_type, entity_text, relation_ids in ner_data:
        unique_ner_labels.add(entity_type)
        while current_idx < begin:
            ner_tags.append((text[current_idx], "O"))
            current_idx += 1
        
        entity_tokens = tokenizer.tokenize(entity_text)
        for i in range(len(entity_tokens)):
            if i == 0:
                ner_tags.append((entity_tokens[i], f"{entity_type}"))
            else:
                ner_tags.append((entity_tokens[i], f"I-{entity_type}"))  # <-- Use "I-" prefix for continuation
            current_idx += 1

        current_idx = end
    
    while current_idx < len(text):
        ner_tags.append((text[current_idx], "O"))
        current_idx += 1
    
    return ner_tags, relation_dict


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
        ner_data, relation_dict = preprocess_ner(json_data, tokenizer)
        preprocessed_ner_data.append(ner_data)
        
        # Preprocess the data for RE tasks
        re_data = preprocess_re(json_data, relation_dict, tokenizer)  # <-- Call preprocess_re
        preprocessed_re_data.append(re_data)  # <-- Store the processed RE data

        print(f"Processed: {file_name}")
        print(f"Number of entities: {len(json_data['entities'])}")
        for entity in json_data['entities']:
            print(entity)

            
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

for ner_data, re_data in tqdm(zip(preprocessed_ner_data, preprocessed_re_data), desc="Tokenizing and aligning labels"):
    print(re_data)
    # Tokenize and align NER labels
    ner_tokens, ner_labels_ = zip(*ner_data)
    encoded_ner = tokenizer.encode_plus(ner_tokens, is_split_into_words=True, add_special_tokens=True, padding="max_length", truncation=True, max_length=512, return_tensors="pt")
    ner_input_ids.append(encoded_ner["input_ids"])
    ner_attention_masks.append(encoded_ner["attention_mask"])

    aligned_ner_labels = []
    for label in ner_labels_:
        label_id = label_to_id[label]
        sub_tokens = tokenizer.tokenize(label.split()[0])
        if label == "O":
            aligned_ner_labels.extend([label_id] * len(sub_tokens))
        else:
            aligned_ner_labels.extend([label_id] + [label_to_id[f"I-{label}"]] * (len(sub_tokens) - 1))



    padded_ner_labels = aligned_ner_labels[:512]
    padded_ner_labels.extend([-100] * (512 - len(padded_ner_labels)))
    ner_labels.append(torch.tensor(padded_ner_labels))

    # Tokenize RE data
    for subject_id, obj_id, relation, obj in re_data:
        subject = json_data["entities"][subject_id]["entityName"]
        obj = json_data["entities"][obj_id]["entityName"]
        tokens = tokenizer.tokenize(f"{subject} [SEP] {obj}")
        encoded_re = tokenizer.encode_plus(tokens, add_special_tokens=True, padding="max_length", truncation=True, max_length=128, return_tensors="pt")

        if encoded_re["input_ids"].shape[1] > 0:
            re_input_ids.append(encoded_re["input_ids"].squeeze(0))
            re_attention_masks.append(encoded_re["attention_mask"].squeeze(0))
            re_labels.append(torch.tensor(relation_to_id[relation]).unsqueeze(0))

ner_input_ids = torch.cat(ner_input_ids, dim=0)
ner_attention_masks = torch.cat(ner_attention_masks, dim=0)
ner_labels = torch.stack(ner_labels, dim=0)

re_input_ids = torch.stack(re_input_ids, dim=0)
re_attention_masks = torch.stack(re_attention_masks, dim=0)
re_labels = torch.cat(re_labels)

print(f"ner_input_ids shape: {ner_input_ids.shape}")
print(f"ner_attention_masks shape: {ner_attention_masks.shape}")
print(f"ner_labels shape: {ner_labels.shape}")

print(f"re_input_ids shape: {re_input_ids.shape}")
print(f"re_attention_masks shape: {re_attention_masks.shape}")
print(f"re_labels shape: {re_labels.shape}")

# Create separate DataLoaders for NER and RE tasks
ner_dataset = TensorDataset(ner_input_ids, ner_attention_masks, ner_labels)
ner_loader = DataLoader(ner_dataset, batch_size=batch_size)

re_dataset = TensorDataset(re_input_ids, re_attention_masks, re_labels)
re_loader = DataLoader(re_dataset, batch_size=batch_size)

# Initialize the custom BERT model
model = BertForNERAndRE.from_pretrained("bert-base-uncased", num_ner_labels=len(label_to_id), num_re_labels=len(relation_to_id))

model.config.num_ner_labels = len(label_to_id)
model.config.num_re_labels = len(relation_to_id)

# Fine-tune the custom BERT model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
total_steps = len(ner_loader) * num_epochs  # You can adjust this based on your requirements

for epoch in range(num_epochs):
    print(f"Epoch {epoch + 1}/{num_epochs}")
    print("-" * 40)

    model.train()
    ner_epoch_loss = 0
    re_epoch_loss = 0
    ner_num_batches = 0
    re_num_batches = 0

    # Training loop for NER and RE combined
    for ner_batch, re_batch in zip(ner_loader, re_loader):
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
        optimizer.zero_grad()
        input_ids, attention_masks, re_labels = tuple(t.to(device) for t in re_batch)
        outputs = model(input_ids, attention_mask=attention_masks, re_labels=re_labels)
        re_loss = outputs[0]
        re_epoch_loss += re_loss.item()
        re_num_batches += 1
        re_loss.backward()
        optimizer.step()

    ner_avg_epoch_loss = ner_epoch_loss / ner_num_batches
    re_avg_epoch_loss = re_epoch_loss / re_num_batches
    print(f"Average NER training loss: {ner_avg_epoch_loss:.4f}")
    print(f"Average RE training loss: {re_avg_epoch_loss:.4f}")


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

