import argparse
import json
import torch
from transformers import BertTokenizer, BertForTokenClassification, BertConfig
from model import BertForNERAndRE

parser = argparse.ArgumentParser(description="Extract relationships from input text")
parser.add_argument("--model_dir", type=str, default="models/combined", help="Directory containing the fine-tuned model and tokenizer")
parser.add_argument("--text", type=str, required=True, help="Input text to extract relationships from")
args = parser.parse_args()

# Load the label_to_id and relation_to_id mappings
with open(f"{args.model_dir}/label_to_id.json", "r") as f:
    label_to_id = json.load(f)

with open(f"{args.model_dir}/relation_to_id.json", "r") as f:
    relation_to_id = json.load(f)

# Load the fine-tuned model and tokenizer
# Initialize the custom BERT model
model = BertForNERAndRE(
    config=BertConfig.from_pretrained(args.model_dir),
    num_ner_labels=len(label_to_id),
    num_re_labels=len(relation_to_id)
)
tokenizer = BertTokenizer.from_pretrained(args.model_dir)

# Tokenize the input text
tokens = tokenizer.tokenize(args.text)
input_ids = tokenizer.convert_tokens_to_ids(tokens)
attention_mask = [1] * len(input_ids)

# Make predictions using the model
with torch.no_grad():
    inputs = {
        "input_ids": torch.tensor([input_ids]),
        "attention_mask": torch.tensor([attention_mask]),
    }
    outputs = model(**inputs)
    ner_predictions = outputs["ner_logits"].argmax(dim=2).tolist()[0]
    re_predictions = outputs["re_logits"].argmax(dim=1).tolist()

# Extract the relationships from the predictions
relationships = []
current_entity = None
current_relation = None

for i, (token, label) in enumerate(zip(tokens, ner_predictions)):
    label = list(label_to_id.keys())[list(label_to_id.values()).index(label)]
    if label.startswith("B-"):
        if current_relation is not None:
            relationships.append((current_entity, current_relation, entity))
        current_entity = token
        current_relation = label.split("-")[1]
    elif label.startswith("I-"):
        if current_entity is None:
            current_entity = token
        else:
            current_entity += " " + token
    else:
        if current_entity is not None and current_relation is not None:
            relationships.append((current_entity, current_relation, entity))
        current_entity = None
        current_relation = None

# Print the extracted relationships
for relationship in relationships:
    print(relationship)
