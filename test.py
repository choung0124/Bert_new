import torch
import logging
import json
import os
from transformers import BertConfig, BertTokenizerFast
from BERT_train import BertForNERAndRE  # Assuming the model is defined in a separate file called 'model.py'

logging.getLogger("transformers").setLevel(logging.ERROR)

output_dir = "models/combined"
with open(os.path.join(output_dir, "label_to_id.json"), "r") as f:
    label_to_id = json.load(f)

with open(os.path.join(output_dir, "relation_to_id.json"), "r") as f:
    relation_to_id = json.load(f)

config = BertConfig.from_pretrained("bert-base-uncased")
tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")

num_ner_labels = len(label_to_id)
num_re_labels = len(relation_to_id)
model = BertForNERAndRE(config, num_ner_labels, num_re_labels)

model_path = "path/to/your/trained/model"  # Replace with the path to your trained model
model.load_state_dict(torch.load(model_path))
model.eval()

def extract_relationships(text, model, tokenizer, id_to_label, id_to_relation):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)

    # Move inputs to the same device as the model
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    # Run the model on the input text
    with torch.no_grad():
        outputs = model(**inputs)

    # Get the predicted NER and RE labels
    ner_predictions = torch.argmax(outputs["ner_logits"], dim=-1).squeeze().tolist()
    re_predictions = torch.argmax(outputs["re_logits"], dim=-1).squeeze().tolist()

    # Convert the predicted labels to their corresponding entity and relationship names
    ner_labels = [id_to_label[str(pred)] for pred in ner_predictions]
    re_labels = [[id_to_relation[str(pred)] for pred in row] for row in re_predictions]

    # Locate and print the subject and object entities along with their relationships
    for i, row in enumerate(re_labels):
        for j, relation in enumerate(row):
            if relation != "no_relation":  # Change this to the name of the "no relation" label in your dataset
                subject = ner_labels[i]
                object = ner_labels[j]
                print(f"{subject} entity: {object} entity: {relation}")

    return ner_labels, re_labels

input_text = "Your input text goes here"
ner_labels, re_labels = extract_relationships(input_text, model, tokenizer, label_to_id, relation_to_id)
