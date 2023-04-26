import torch
from transformers import BertTokenizer, BertForNERAndRE
import json
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--text", type=str, required=True, help="Input text to extract relationships from")
args = parser.parse_args()

# Load the fine-tuned custom BERT model and tokenizer
output_dir = "models/combined"
tokenizer = BertTokenizer.from_pretrained(output_dir)
model = BertForNERAndRE.from_pretrained(output_dir)

# Load the label_to_id and relation_to_id mappings
with open(os.path.join(output_dir, "label_to_id.json"), "r") as f:
    label_to_id = json.load(f)

with open(os.path.join(output_dir, "relation_to_id.json"), "r") as f:
    relation_to_id = json.load(f)

# Function to extract relationships from input text
def extract_relationships(input_text):
    input_tokens = tokenizer.encode_plus(input_text, add_special_tokens=True, padding="max_length", truncation=True, max_length=512, return_tensors="pt")
    input_ids = input_tokens["input_ids"].to(device)
    attention_mask = input_tokens["attention_mask"].to(device)

    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)

    ner_logits, re_logits = outputs[1], outputs[2]
    ner_predictions = torch.argmax(ner_logits, dim=2)[0]
    re_predictions = torch.argmax(re_logits, dim=1)[0]

    # Extract the subject and object entities
    entities = []
    entity_text = ""
    is_entity = False

    for i, (token_id, ner_label_id) in enumerate(zip(input_ids.squeeze(), ner_predictions)):
        token = tokenizer.convert_ids_to_tokens(token_id.item())
        ner_label = list(label_to_id.keys())[list(label_to_id.values()).index(ner_label_id.item())]

        if not is_entity and ner_label.startswith("B-"):
            is_entity = True
            entity_text = token.replace("##", "")
        elif is_entity and ner_label.startswith("I-"):
            entity_text += token.replace("##", "")
        elif is_entity and not ner_label.startswith("I-"):
            is_entity = False
            entities.append((entity_text, ner_label.split("-")[1]))

    # Extract the relations between the subject and object entities
    relationships = []
    relation_text = ""
    is_relation = False
    subject, obj = None, None

    for i, (token_id, re_label_id) in enumerate(zip(input_ids.squeeze(), re_predictions)):
        token = tokenizer.convert_ids_to_tokens(token_id.item())
        re_label = list(relation_to_id.keys())[list(relation_to_id.values()).index(re_label_id.item())]

        if not is_relation and re_label != "NA":
            is_relation = True
            relation_text = re_label
            subject = entities[i][0]
        elif is_relation and re_label != "NA":
            relation_text = re_label
        elif is_relation and re_label == "NA":
            is_relation = False
            obj = entities[i][0]
            relationships.append((subject, relation_text, obj))

    return relationships

# Example usage
input_text = args.text
relationships = extract_relationships(input_text)

print("Extracted relationships:")
for subject, relationship, obj in relationships:
    print(f"{subject} - {relationship} - {obj}")
