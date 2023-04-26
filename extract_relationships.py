import torch
import json
import argparse
from transformers import BertTokenizer
from BERT_full_train import BertForNERAndRE

parser = argparse.ArgumentParser(description="Extract relationships from text using a fine-tuned BERT model")
parser.add_argument("text", type=str, help="the input text")
args = parser.parse_args()

tokenizer = BertTokenizer.from_pretrained("models/combined")

with open("models/combined/label_to_id.json", "r") as f:
    label_to_id = json.load(f)

with open("models/combined/relation_to_id.json", "r") as f:
    relation_to_id = json.load(f)

# Initialize the custom BERT model
num_ner = len(label_to_id)
num_re = len(relation_to_id)
model = BertForNERAndRE.from_pretrained("models/combined", num_ner_labels=num_ner, num_re_labels=num_re)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

model.eval()

tokens = tokenizer.encode_plus(args.text, is_split_into_words=True, add_special_tokens=True, padding="max_length", truncation=True, max_length=512, return_tensors="pt")

input_ids = tokens["input_ids"].to(device)
attention_mask = tokens["attention_mask"].to(device)

ner_logits, re_logits = model(input_ids, attention_mask=attention_mask)

ner_preds = torch.argmax(ner_logits, axis=-1)
re_preds = torch.argmax(re_logits, axis=-1)

ner_tags = [list(label_to_id.keys())[label_id] for label_id in ner_preds[0].tolist()]
re_label = list(relation_to_id.keys())[re_preds[0].item()]

subject_start = -1
subject_end = -1
object_start = -1
object_end = -1
subject = ""
object = ""
inside_entity = False
for i, (token, ner_tag) in enumerate(zip(tokenizer.convert_ids_to_tokens(input_ids[0]), ner_tags)):
    if not inside_entity and ner_tag.startswith("B"):
        if ner_tag[2:] == "SUB":
            subject_start = i
        elif ner_tag[2:] == "OBJ":
            object_start = i
        inside_entity = True
    elif inside_entity and not ner_tag.startswith("I"):
        if ner_tag.startswith("B"):
            if ner_tag[2:] == "SUB":
                subject_start = i
            elif ner_tag[2:] == "OBJ":
                object_start = i
            inside_entity = True
        else:
            if ner_tag == "O":
                if subject_start != -1 and subject_end == -1:
                    subject_end = i
                elif object_start != -1 and object_end == -1:
                    object_end = i
                inside_entity = False

if subject_start != -1 and subject_end == -1:
    subject_end = len(ner_tags)

if object_start != -1 and object_end == -1:
    object_end = len(ner_tags)

if subject_start != -1 and subject_end != -1:
    subject = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(input_ids[0][subject_start:subject_end]))

if object_start != -1 and object_end != -1:
    object = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(input_ids[0][object_start:object_end]))

if subject != "" and object != "":
    print(f"{subject} {re_label} {object}")
else:
    print("No relationship found in the input text.")
