import torch
from transformers import BertTokenizer, BertForSequenceClassification
import json
import sys


# Load the pretrained model and tokenizer
model_path = "models/combined/" # Change this to the path of your pretrained model directory
tokenizer = BertTokenizer.from_pretrained(model_path)
model = BertForSequenceClassification.from_pretrained(model_path)

# Set the device (CPU or GPU) to use for inference
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Define a function to extract relationships from input text
def extract_relationships(text):
    # Tokenize the input text
    tokens = tokenizer.encode_plus(text, add_special_tokens=True, return_tensors="pt")
    input_ids = tokens["input_ids"].to(device)
    attention_mask = tokens["attention_mask"].to(device)

    # Make predictions with the model
    outputs = model(input_ids, attention_mask=attention_mask)
    _, predicted = torch.max(outputs[0], dim=1)

    # Map the predicted labels back to relation names using the relation_to_id mapping
    with open(f"{model_path}/relation_to_id.json", "r") as f:
        relation_to_id = json.load(f)
        id_to_relation = {v: k for k, v in relation_to_id.items()}
        predicted_relations = [id_to_relation[label.item()] for label in predicted]

    # Extract subject-object pairs from the input text and the predicted relations
    subject, object_ = None, None
    relationships = []
    for token, relation in zip(tokenizer.tokenize(text), predicted_relations):
        if relation == "SUBJECT":
            subject = token
        elif relation == "OBJECT":
            object_ = token
        elif relation != "O":
            relationships.append((subject, relation, object_))
            subject, object_ = None, None

    # Return the extracted relationships
    return relationships

# Get the input text from the command line argument
text = sys.argv[1]

# Extract the relationships from the input text
relationships = extract_relationships(text)

# Print the extracted relationships
print("Extracted relationships:")
for subject, relationship, obj in relationships:
    print(f"{subject} - {relationship} - {obj}")
