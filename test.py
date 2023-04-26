import argparse
from predict import predict_ner, predict_re

parser = argparse.ArgumentParser(description='Predict entities and relations from text.')
parser.add_argument('text', type=str, help='input text')

args = parser.parse_args()

text = args.text

entities = predict_ner(text)
relations = predict_re(text, entities)

print("Entities:")
print(entities)
print("\nRelations:")
print(relations)
