import argparse
from predict import tokenizer, model, id_to_relation, predict_ner, predict_re
import json
import logging
from tqdm import tqdm

logging.getLogger("transformers").setLevel(logging.ERROR)

parser = argparse.ArgumentParser(description='Predict entities and relations from text.')
parser.add_argument('text', type=str, help='input text')

args = parser.parse_args()

text = args.text

with open("models/combined/relation_to_id.json", "r") as f:
    relation_to_id = json.load(f)

with open("models/combined/label_to_id.json", "r") as f:
    label_to_id = json.load(f)

print("Predicting entities...")
entities = predict_ner(text)
print(f"Found {len(entities)} entities:")
for entity in entities:
    print(entity)

print("Predicting relations...")
relations = []
for i, entity1 in tqdm(enumerate(entities), desc="Entity pairs"):
    for j, entity2 in tqdm(enumerate(entities), desc="Entity pairs"):
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
            relation_name = id_to_relation[prediction]
            relation_data = {"subjectId": subject_id, "objectId": object_id, "subjectName": subject, "objectName": obj, "relationName": relation_name}
            relations.append(relation_data)
print(f"Found {len(relations)} relations:")
for relation in relations:
    print(relation)
