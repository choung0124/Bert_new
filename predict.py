import os
import torch
import json
from transformers import BertTokenizer, BertModel
from typing import List
import logging
logging.getLogger("transformers").setLevel(logging.ERROR)

tokenizer = BertTokenizer.from_pretrained("models/combined")
model = BertModel.from_pretrained("models/combined")
model.eval()

with open("models/combined/label_to_id.json", "r") as f:
    label_to_id = json.load(f)

with open("models/combined/relation_to_id.json", "r") as f:
    relation_to_id = json.load(f)

id_to_label = {v: k for k, v in label_to_id.items()}
id_to_relation = {v: k for k, v in relation_to_id.items()}


def predict_ner(text: str, confidence_threshold: float = 0.5) -> List[dict]:
    with open("models/combined/label_to_id.json", "r") as f:
        label_to_id = json.load(f)
    
    # Load the id_to_label dictionary using the label_to_id dictionary
    id_to_label = {v: k for k, v in label_to_id.items()}
    
    input_ids = tokenizer.encode(text, add_special_tokens=True, return_tensors="pt")
    with torch.no_grad():
        outputs = model(input_ids)
    predictions = outputs.last_hidden_state.argmax(-1).tolist()[0]
    scores = torch.softmax(outputs.last_hidden_state, dim=2).max(dim=2).values.tolist()[0]

    tokens = tokenizer.convert_ids_to_tokens(input_ids.tolist()[0])

    entities = []
    current_entity = {"entityName": "", "entityType": "", "entityId": "", "score": 0.0}
    for i, (token, prediction, score) in enumerate(zip(tokens, predictions, scores)):
        if score < confidence_threshold:
            continue
        if token.startswith("##"):
            current_entity["entityName"] += token[2:]
        else:
            if current_entity["entityType"]:
                entities.append(current_entity.copy())
            current_entity["entityName"] = token
            current_entity["entityType"] = id_to_label.get(prediction, "O")
            current_entity["entityId"] = f"T{i}"
            current_entity["score"] = score
    if current_entity["entityType"]:
        entities.append(current_entity.copy())
    return entities




def predict_re(text: str, entities: List[dict]) -> List[dict]:
    with open("models/combined/relation_to_id.json", "r") as f:
        relation_to_id = json.load(f)

    # Load the id_to_relation dictionary using the relation_to_id dictionary
    id_to_relation = {v: k for k, v in relation_to_id.items()}

    relation_data = []
    for i, entity1 in enumerate(entities):
        for j, entity2 in enumerate(entities):
            if i != j:
                subject_id = entity1["entityId"]
                object_id = entity2["entityId"]
                subject = entity1["entityName"]
                obj = entity2["entityName"]
                relation_name = ""
                input_text = f"{subject} [SEP] {obj}"
                input_ids = tokenizer.encode(input_text, add_special_tokens=True, return_tensors="pt")
                with torch.no_grad():
                    outputs = model(input_ids)
                prediction = outputs.pooler_output.argmax(-1).item()
                relation_name = id_to_relation.get(prediction, "UNKNOWN")
                relation_data.append({"subjectId": subject_id, "objectId": object_id, "subjectName": subject, "objectName": obj, "relationName": relation_name})
    return relation_data



