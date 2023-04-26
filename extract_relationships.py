import os
import sys
import json
import torch
from torch.utils.data import DataLoader, TensorDataset
from transformers import BertTokenizer, BertModel, BertPreTrainedModel
from torch import nn

from BERT_full_train import BertForNERAndRE, preprocess_ner, preprocess_re

# Load the model, tokenizer, label mappings
output_dir = "models/combined"
model = BertForNERAndRE.from_pretrained(output_dir)
tokenizer = BertTokenizer.from_pretrained(output_dir)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

with open(os.path.join(output_dir, "models/combined/label_to_id.json"), "r") as f:
    label_to_id = json.load(f)
    id_to_label = {idx: label for label, idx in label_to_id.items()}

with open(os.path.join(output_dir, "models/combined/relation_to_id.json"), "r") as f:
    relation_to_id = json.load(f)
    id_to_relation = {idx: relation for relation, idx in relation_to_id.items()}

# Input text from command line argument
input_text = sys.argv[1]

# Preprocess and tokenize input text
json_data = {"text": input_text, "entities": [], "relation_info": []}
ner_data = preprocess_ner(json_data, tokenizer)
re_data = preprocess_re(json_data)

# Prepare input for the model
ner_tokens, ner_labels_ = zip(*ner_data)
encoded_ner = tokenizer.encode_plus(ner_tokens, is_split_into_words=True, add_special_tokens=True, padding="max_length", truncation=True, max_length=512, return_tensors="pt")
input_ids = encoded_ner["input_ids"].to(device)
attention_mask = encoded_ner["attention_mask"].to(device)

# Run model predictions
with torch.no_grad():
    ner_logits, re_logits = model(input_ids, attention_mask=attention_mask)

ner_logits = ner_logits.cpu().detach().numpy()
re_logits = re_logits.cpu().detach().numpy()

# Extract NER and RE predictions
ner_pred = [id_to_label[lbl] for lbl in ner_logits.argmax(axis=-1).flatten().tolist()]
re_pred = [id_to_relation[lbl] for lbl in re_logits.argmax(axis=-1).flatten().tolist()]

# Print NER and RE predictions
print("NER Predictions:")
for token, label in zip(ner_tokens, ner_pred):
    print(f"{token}: {label}")

print("\nRE Predictions:")
for relation in re_pred:
    print(relation)
