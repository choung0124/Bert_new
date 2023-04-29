import torch
import logging
import pickle
import os
from transformers import DistilBertConfig, DistilBertTokenizerFast
from DistiliBERT_train import DistilBertForNERAndRE  # Assuming the model is defined in a separate file called 'model.py'

logging.getLogger("transformers").setLevel(logging.ERROR)

output_dir = "models/combined"
with open(os.path.join(output_dir, "label_to_id.pkl"), "rb") as f:
    label_to_id = pickle.load(f)

with open(os.path.join(output_dir, "relation_to_id.pkl"), "rb") as f:
    relation_to_id = pickle.load(f)

config = DistilBertConfig.from_pretrained("distilbert-base-uncased")
tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")

num_ner_labels = len(label_to_id)
num_re_labels = len(relation_to_id)
model = DistilBertForNERAndRE(config, num_ner_labels, num_re_labels)

model_path = "models/combined/pytorch_model.bin"  # Replace with the path to your trained model
model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
model.eval()

# Move the model to the appropriate device (GPU if available, otherwise CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
def extract_relationships(text, model, tokenizer, id_to_label, id_to_relation):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)

    # Move inputs to the same device as the model
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    # Run the model on the input text
    with torch.no_grad():
        outputs = model(**inputs)

    # Check if re_logits is not None
    if outputs["re_logits"] is None:
        print("Error: re_logits is None")
        return None, None

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

input_text = "Sildenafil treats MI"
ner_labels, re_labels = extract_relationships(input_text, model, tokenizer, label_to_id, relation_to_id)
