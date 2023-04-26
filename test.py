import argparse
from predict import predict_ner, predict_re

parser = argparse.ArgumentParser(description='Predict entities and relations from text.')
parser.add_argument('text', type=str, help='input text')

args = parser.parse_args()

text = args.text

with open("models/combined/relation_to_id.json", "r") as f:
    relation_to_id = json.load(f)
    
with open("models/combined/label_to_id.json", "r") as f:
    label_to_id = json.load(f)


entities = predict_ner(text)
relations = predict_re(text, entities)

print("Entities:")
print(entities)
print("\nRelations:")
print(relations)
