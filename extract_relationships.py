import argparse
import json
import torch
from transformers import BertTokenizer
from model import BertForNERAndRE

# Define command line arguments
parser = argparse.ArgumentParser(description="Extract relationships from text using a fine-tuned BERT model.")
parser.add_argument("text", type=str, help="Input text to extract relationships from")
parser.add_argument("model_dir", type=str, help="Directory containing the fine-tuned BERT model and tokenizer")
parser.add_argument("--device", type=str, default="cpu", help="Device to run the model on. Defaults to 'cpu'")

# Parse command line arguments
args = parser.parse_args()

# Load the model and tokenizer
model = BertForNERAndRE.from_pretrained(args.model_dir)
tokenizer = BertTokenizer.from_pretrained(args.model_dir)

# Set the device to run the model on
device = torch.device(args.device)
model.to(device)

# Tokenize the input text
tokens = tokenizer.encode_plus(args.text, add_special_tokens=True, padding="max_length", truncation=True, max_length=512, return_tensors="pt")
input_ids = tokens["input_ids"].to(device)
attention_mask = tokens["attention_mask"].to(device)

# Make predictions using the model
ner_logits, re_logits = model(input_ids=input_ids, attention_mask=attention_mask)
ner_predictions = torch.argmax(ner_logits, dim=2)
re_predictions = torch.argmax(re_logits, dim=1)

# Decode the predictions
ner_labels = tokenizer.convert_ids_to_tokens(ner_predictions[0])
re_label = next(key for key, value in model.relation_to_id.items() if value == re_predictions[0].item())

# Extract relationships from the NER labels and RE relation label
relationships = []
current_subject = None
current_relation = None
for i in range(len(ner_labels)):
    label = ner_labels[i]
    if label.startswith("B-"):
        entity_type = label.split("-")[1]
        current_subject = tokenizer.convert_tokens_to_string([tokenizer.ids_to_tokens(input_ids[0][i].item())])
        current_relation = None
    elif label.startswith("I-"):
        entity_type = label.split("-")[1]
        current_subject += " " + tokenizer.convert_tokens_to_string([tokenizer.ids_to_tokens(input_ids[0][i].item())])
    else:
        entity_type = "O"
        if current_subject is not None:
            if current_relation is None and entity_type == "O":
                continue
            if current_relation is None:
                current_relation = current_subject + " " + re_label
            else:
                obj = tokenizer.convert_tokens_to_string([tokenizer.ids_to_tokens(input_ids[0][i-1].item())])
                current_relation += " " + obj
                relationships.append(current_relation)
                current_relation = None
            current_subject = None

# Print the extracted relationships
if len(relationships) == 0:
    print("No relationships found in the input text.")
else:
    print("Extracted relationships:")
    for relationship in relationships:
        print(relationship)
