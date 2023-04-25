import json
import os
import re

def generate_label_dicts(dir_path):
    subject_labels = set()
    object_labels = set()
    re_labels = set()

    for file_name in os.listdir(dir_path):
        if file_name.endswith('.json'):
            with open(os.path.join(dir_path, file_name), 'r') as file:
                content = file.read()
                data = json.loads(content)
                entities = data['entities']
                relations = data['relation_info']

            for relation in relations:
                subject_entity = next(entity for entity in entities if entity['entityId'] == relation['subjectID'])
                object_entity = next(entity for entity in entities if entity['entityId'] == relation['objectId'])

                subject_labels.add(subject_entity['entityType'])
                object_labels.add(object_entity['entityType'])
                re_labels.add(relation['rel_name'])

    subject_label2idx = {label: idx for idx, label in enumerate(subject_labels, start=1)}
    object_label2idx = {label: idx for idx, label in enumerate(object_labels, start=1)}
    re_label2idx = {label: idx for idx, label in enumerate(re_labels, start=1)}

    # Add "O" (Outside) label for NER
    ner_label2idx = {
        "O": 0,
        **subject_label2idx,
        **object_label2idx,
    }

    return subject_label2idx, object_label2idx, re_label2idx, ner_label2idx

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--dir_path', type=str, required=True, help='Path to the directory containing the JSON files')
    args = parser.parse_args()

    subject_label2idx, object_label2idx, re_label2idx, ner_label2idx = generate_label_dicts(args.dir_path)
