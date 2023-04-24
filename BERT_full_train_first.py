import json
from torch.utils.data import Dataset
from transformers import BertTokenizerFast
import torch
from torch import nn
from transformers import BertModel
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import sys
import re

if len(sys.argv) < 2:
    print("Usage: python script_name.py <directory_name>")
    sys.exit(1)

# set the directory path where the JSON files are located
dir_path = sys.argv[1]

# create lists to store the results
full_text = []
full_entities = []
entity_sentences = []
full_relations = []
relation_sentences = []

# define a regular expression pattern to match sentence boundaries
pattern = re.compile(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s')

# loop through each file in the directory
for file_name in os.listdir(dir_path):
    if file_name.endswith('.json'):
        # read the JSON file and extract the content field from each object
        with open(os.path.join(dir_path, file_name), 'r') as file:
            content = file.read()
            data = json.loads(content)
            text = data['text']
            entities = data['entities']
            relations = data['relation_info']
        full_text.append(text)
        full_entities.append(entities)
        full_relations.append(relations)

        # find the sentence containing each entity
        for entity in entities:
            span_begin = entity['span']['begin']
            span_end = entity['span']['end']
            sentences = pattern.split(text)
            for sentence in sentences:
                if text.find(sentence) <= span_begin and text.find(sentence) + len(sentence) >= span_end:
                    entity_sentences.append({'entity': entity['entityName'], 'sentence': sentence})

        # find the sentence containing each relation
        for relation in relations:
            subject_text = relation['subjectText']
            object_text = relation['objectText']
            if subject_text in text and object_text in text:
                sentences = pattern.split(text)
                for sentence in sentences:
                    if subject_text in sentence and object_text in sentence:
                        relation_sentences.append({'relation': relation['rel_name'], 'sentence': sentence})

# write the entity sentences to a text file
try:
    with open('entity_sentences.txt', 'w') as file:
        for entity_sentence in entity_sentences:
            file.write(f"Entity: {entity_sentence['entity']}\n")
            file.write(f"Sentence: {entity_sentence['sentence']}\n\n")
except Exception as e:
    print(f"Error writing entity sentences: {e}")

# write the relation sentences to a text file
try:
    with open('relation_sentences.txt', 'w') as file:
        for relation_sentence in relation_sentences:
            file.write(f"Relation: {relation_sentence['relation']}\n")
            file.write(f"Sentence: {relation_sentence['sentence']}\n\n")
except Exception as e:
    print(f"Error writing relation sentences: {e}")

# print the number of entity and relation sentences found
print(f"Found {len(entity_sentences)} entity sentences")
print(f"Found {len(relation_sentences)} relation sentences")

sentences = [item['sentence'] for item in entity_sentences + relation_sentences]
ner_labels = [item['entity'] for item in entity_sentences] + [None] * len(relation_sentences)
re_labels = [None] * len(entity_sentences) + [item['relation'] for item in relation_sentences]

# Create label-to-index mapping dictionaries for NER and RE tasks
ner_label2idx = {label: idx for idx, label in enumerate(set(ner_labels))}
re_label2idx = {label: idx for idx, label in enumerate(set(re_labels))}

# Convert NER and RE labels to indices
ner_labels_idx = [ner_label2idx[label] if label is not None else None for label in ner_labels]
re_labels_idx = [re_label2idx[label] if label is not None else None for label in re_labels]

# Create index-to-label mapping dictionaries for NER and RE tasks
idx2ner_label = {idx: label for label, idx in ner_label2idx.items()}
idx2re_label = {idx: label for label, idx in re_label2idx.items()}

# Convert NER and RE label indices back to their original labels
ner_labels_original = [idx2ner_label[idx] if idx is not None else None for idx in ner_labels_idx]
re_labels_original = [idx2re_label[idx] if idx is not None else None for idx in re_labels_idx]

# Create label-to-index mapping dictionaries for NER and RE tasks
ner_label2idx = {label: idx for idx, label in enumerate(set(ner_labels))}
re_label2idx = {label: idx for idx, label in enumerate(set(re_labels))}
ner_label2idx[None] = len(ner_label2idx) - 1
re_label2idx[None] = len(re_label2idx) - 1

class NERRE_Dataset(Dataset):
    def __init__(self, sentences, ner_labels, re_labels):
        self.sentences = sentences
        self.ner_labels = ner_labels
        self.re_labels = re_labels

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        return self.sentences[idx], self.ner_labels[idx], self.re_labels[idx]

tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')

def tokenize_data(dataset):
    tokenized_data = []
    for sentence, ner_label, re_label in dataset:
        tokens = tokenizer(sentence, truncation=True, padding='max_length', max_length=128, return_tensors='pt')
        ner_label_tensor = torch.tensor(ner_label if ner_label is not None else ner_label2idx[None], dtype=torch.long)
        re_label_tensor = torch.tensor(re_label if re_label is not None else re_label2idx[None], dtype=torch.long)
        tokenized_data.append((tokens, ner_label_tensor, re_label_tensor))
    return tokenized_data

class NER_RE_Model(nn.Module):
    def __init__(self, ner_label_count, re_label_count):
        super(NER_RE_Model, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.ner_classifier = nn.Linear(self.bert.config.hidden_size, ner_label_count)
        self.re_classifier = nn.Linear(self.bert.config.hidden_size, re_label_count)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        ner_logits = self.ner_classifier(pooled_output)
        re_logits = self.re_classifier(pooled_output)
        return ner_logits, re_logits

def create_data_loader(tokenized_data, batch_size):
    dataset = [(item[0]['input_ids'].squeeze(0), item[0]['attention_mask'].squeeze(0), item[1], item[2]) for item in tokenized_data]
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)

def train_epoch(model, data_loader, optimizer, device):
    model.train()
    for batch in tqdm(data_loader):
        input_ids, attention_mask, ner_label, re_label = batch
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        ner_labels = ner_label.to(device)  # This line is changed
        re_labels = re_label.to(device)  # This line is changed

        optimizer.zero_grad()
        ner_logits, re_logits = model(input_ids, attention_mask)
        loss_fn = nn.CrossEntropyLoss()
        ner_loss = loss_fn(ner_logits, ner_labels)
        re_loss = loss_fn(re_logits, re_labels)
        total_loss = ner_loss + re_loss
        total_loss.backward()
        optimizer.step()


# Filter out None values
filtered_data = [(sentence, ner_label, re_label) for sentence, ner_label, re_label in zip(sentences, ner_labels_idx, re_labels_idx) if sentence is not None]

# Create the train_dataset with filtered data
train_dataset = NERRE_Dataset(*zip(*filtered_data))

tokenized_train_data = tokenize_data(train_dataset)
train_data_loader = DataLoader(tokenized_train_data, batch_size=8, shuffle=True)

model = NER_RE_Model(len(ner_label2idx), len(re_label2idx))

batch_size = 8
train_data_loader = create_data_loader(tokenized_train_data, batch_size)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

optimizer = AdamW(model.parameters(), lr=3e-5)

num_epochs = 3

for epoch in range(num_epochs):
    print(f"Epoch {epoch+1}/{num_epochs}")
    train_epoch(model, train_data_loader, optimizer, device)

model_dir = "model"
model.bert.save_pretrained(model_dir)

# Save the custom head
torch.save({
    'ner_classifier': model.ner_classifier.state_dict(),
    're_classifier': model.re_classifier.state_dict()
}, os.path.join(model_dir, 'custom_head.pt'))

# Save the tokenizer
tokenizer.save_pretrained(model_dir)