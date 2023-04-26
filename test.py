import argparse
import re
from entity_relation_extraction import extract_entities, extract_relations

# Define a regular expression pattern to match relation phrases between two entities
relation_pattern = re.compile(r"\[(?P<subject>e\d+)\]\s(?P<relation>.+)\s\[(?P<object>e\d+)\]")

if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("text", help="input text to extract entities and relations from")
    args = parser.parse_args()

    # Extract entities from input text
    text, entities = extract_entities(args.text)

    # Extract relations from input text and identified entities
    relations = extract_relations(text, entities)

    # Print identified entities and relations
    print("Entities:")
    for entity_id, entity in entities.items():
        print(f"- {entity['type']}: {entity['name']} ({entity_id})")
    print("Relations:")
    for relation in relations:
        print(f"- {relation['name']}: {relation['subject']['name']} ({relation['subject']['type']}) -> {relation['object']['name']} ({relation['object']['type']})")
