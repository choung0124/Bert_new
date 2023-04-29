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
import spacy

#os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:64'
os.environ["TOKENIZERS_PARALLELISM"] = "false"
tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")
#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu')
batch_size = 8
num_epochs = 4
learning_rate = 5e-5

unique_ner_labels = set()
unique_relation_labels = set()
unique_ner_labels.add("O")

# Existing preprocessing functions
import itertools
import spacy

def preprocess_data(json_data, tokenizer, label_to_id, relation_to_id):
    ner_data = []
    re_data = []

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

    # Load spaCy's small English model for sentence tokenization
    nlp = spacy.load("en_core_web_sm", disable=["tagger", "parser", "ner", "lemmatizer", "attribute_ruler"])
    nlp.add_pipe('sentencizer')
    doc = nlp(text)

    # Break the text into sentences
    sentences = [sent.text for sent in doc.sents]

    # Find sentence boundaries
    sentence_boundaries = [0]
    for sentence in sentences[:-1]:
        sentence_boundaries.append(sentence_boundaries[-1] + len(sentence) + 1)

    # Filter sentences containing entities
    relevant_sentences = []
    for i, (sentence, boundary) in enumerate(zip(sentences, sentence_boundaries)):
        if any(boundary <= entity["span"]["begin"] < boundary + len(sentence) for entity in json_data["entities"]):
            relevant_sentences.append((sentence, i, boundary))

    # Process entities
    for entity in sorted(json_data["entities"], key=lambda x: x["span"]["begin"]):
        begin, end = entity["span"]["begin"], entity["span"]["end"]
        entity_type = entity["entityType"]
        entity_id = entity["entityId"]
        entity_name = entity["entityName"]

        # Find the relevant sentence index containing the entity
        sentence_idx, boundary = next((i, boundary) for sentence, i, boundary in relevant_sentences if boundary + len(sentence) >= begin)

        # Tokenize the relevant sentence
        sentence_doc = nlp(relevant_sentences[sentence_idx][0])
        sentence_tokens = [token for token in sentence_doc]

        # Find the token index of the entity
        entity_start_idx = next(i for i, token in enumerate(sentence_tokens) if token.idx == begin - boundary)
        entity_end_idx = next(i for i, token in enumerate(sentence_tokens) if token.idx == end - boundary) - 1

        # Annotate the tokens with the entity label
        for i, token in enumerate(sentence_tokens):
            if i == entity_start_idx:
                label = f"B-{entity_type}-{entity_name}"
            elif entity_start_idx < i <= entity_end_idx:
                label = f"I-{entity_type}-{entity_name}"
            else:
                label = "O"

            if label not in label_to_id:
                label_to_id[label] = len(label_to_id)

            ner_data.append((token.text, label, len(ner_data)))

        if f"{entity_type}-{entity_name}" not in label_to_id:
            label_to_id[f"{entity_type}-{entity_name}"] = len(label_to_id)

    # Process relations
    for entity_id_1, entity_id_2 in itertools.combinations(entity_ids, 2):
        if entity_id_1 in relation_dict and entity_id_2 in relation_dict[entity_id_1]:
            rel_name = relation_dict[entity_id_1][entity_id_2]
            entity_1 = entities_dict[entity_id_1]
            entity_2 = entities_dict[entity_id_2]

            # Find the relevant sentence index containing the entities
            sentence_idx, boundary = next((i, boundary) for sentence, i, boundary in relevant_sentences if boundary + len(sentence) >= entity_1["span"]["begin"])

            re_data.append({
                'id': (entity_id_1, entity_id_2),
                'subject': relevant_sentences[sentence_idx][0][entity_1["span"]["begin"] - boundary:entity_1["span"]["end"] - boundary],
                'object': relevant_sentences[sentence_idx][0][entity_2["span"]["begin"] - boundary:entity_2["span"]["end"] - boundary],
                'relation': rel_name,
                'subject_tokens': tokenizer.tokenize(relevant_sentences[sentence_idx][0][entity_1["span"]["begin"] - boundary:entity_1["span"]["end"] - boundary]),
                'object_tokens': tokenizer.tokenize(relevant_sentences[sentence_idx][0][entity_2["span"]["begin"] - boundary:entity_2["span"]["end"] - boundary])
            })

            if rel_name not in relation_to_id:
                relation_to_id[rel_name] = len(relation_to_id)

    return ner_data, re_data, label_to_id, relation_to_id


class NERRE_Dataset(Dataset):
    def __init__(self, ner_data, re_data, tokenizer, max_length, label_to_id, relation_to_id):
        self.ner_data = ner_data
        self.re_data = re_data
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.label_to_id = label_to_id
        self.relation_to_id = relation_to_id

    def __len__(self):
        return len(self.ner_data)

    def __getitem__(self, idx):
        item = self.ner_data[idx]
        tokens = [token for token, label, _ in item]
        ner_labels = [label for _, label, _ in item]
        ner_label_ids = [self.label_to_id[label] for label in ner_labels]

        inputs = self.tokenizer(tokens, padding='max_length', truncation=True, max_length=self.max_length, return_tensors='pt')

        input_ids = inputs['input_ids'].squeeze()
        attention_mask = inputs['attention_mask'].squeeze()

        re_item = self.re_data[idx]
        re_labels = [self.relation_to_id[relation['relation']] for relation in re_item]

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'ner_labels': torch.tensor(ner_label_ids, dtype=torch.long),
            're_labels': torch.tensor(re_labels, dtype=torch.long),
            're_data': re_item
        }

def custom_collate_fn(batch):
    # Remove None values from the batch
    batch = [item for item in batch if item is not None]

    # Pad input_ids, attention_mask, and token_type_ids
    input_ids = pad_sequence([item['input_ids'] for item in batch], batch_first=True)
    attention_mask = pad_sequence([item['attention_mask'] for item in batch], batch_first=True)

    # Pad ner_labels
    ner_labels = pad_sequence([item['ner_labels'] for item in batch], batch_first=True, padding_value=-100)
    
    # Pad re_labels
    re_labels = pad_sequence([item['re_labels'] for item in batch], batch_first=True, padding_value=-1)

    # Return the final dictionary
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "ner_labels": ner_labels,
        "re_labels": re_labels,
        "re_data": [item['re_data'] for item in batch],
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
    num_workers = 2
else:
    num_workers = 6
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
        head_mask=None,
        inputs_embeds=None,
        ner_labels=None,
        re_labels=None,
        re_data=None,
    ):
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
            loss_fct = nn.CrossEntropyLoss(ignore_index=-1)
            active_loss = attention_mask.view(-1) == 1
            active_logits = ner_logits.view(-1, self.num_ner_labels)[active_loss]
            active_labels = ner_labels.view(-1)[active_loss]
            ner_loss = loss_fct(active_logits, active_labels)
        else:
            ner_loss = None

        if re_data is not None and len(re_data) > 0:
            re_logits = []
            for b, batch_re_data in enumerate(re_data):
                batch_re_logits = []
                for rel in batch_re_data:
                    subject_idx = rel["subject_idx"]
                    object_idx = rel["object_idx"]

                    subject_hidden_states = sequence_output[b, subject_idx]
                    object_hidden_states = sequence_output[b, object_idx]

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
accumulation_steps = 32  # Increase this value based on the desired accumulation steps.
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

            input_ids = batch['input_ids'].view(-1, batch['input_ids'].size(-1)).to(device)
            attention_mask = batch['attention_mask'].view(-1, batch['attention_mask'].size(-1)).to(device)
            ner_labels = batch['ner_labels'].view(-1).to(device)
            re_labels = batch['re_labels'].to(device)
            re_data = batch['re_data']

            if scaler is not None:
                with autocast():
                    outputs = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        ner_labels=ner_labels,
                        re_labels=re_labels,
                        re_data=re_data,
                    )
            else:
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    ner_labels=ner_labels,
                    re_labels=re_labels,
                    re_data=re_data,
                )

            # Get NER and RE logits from model output
            ner_logits = outputs["ner_logits"]
            re_logits = outputs["re_logits"]

            # Calculate NER loss
            ner_loss = ner_loss_fn(ner_logits.view(-1, ner_logits.size(-1)), ner_labels.view(-1))

            # Calculate RE loss
            re_loss = 0
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
