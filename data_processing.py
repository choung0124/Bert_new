import json
import os
import re

def read_json_files(dir_path):
    full_text = []
    full_entities = []
    full_relations = []

    pattern = re.compile(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s')

    for file_name in os.listdir(dir_path):
        if file_name.endswith('.json'):
            with open(os.path.join(dir_path, file_name), 'r') as file:
                content = file.read()
                data = json.loads(content)
                text = data['text']
                entities = data['entities']
                relations = data['relation_info']
            full_text.append(text)
            full_entities.append(entities)
            full_relations.append(relations)

    return full_text, full_entities, full_relations

def extract_sentences(full_text, full_entities, full_relations):
    entity_sentences = []
    relation_sentences = []

    for text, entities, relations in zip(full_text, full_entities, full_relations):
        for entity in entities:
            start = entity["span"]["begin"]
            end = entity["span"]["end"]
            entity_sentence = text[max(0, start - 10):min(end + 10, len(text))]
            entity_sentences.append({"sentence": entity_sentence, "entity": entity["Type"]})

        for relation in relations:
            subject_start, subject_end = map(int, relation["subjectID"].split("_"))
            object_start, object_end = map(int, relation["objectId"].split("_"))

            relation_sentence = text[min(subject_start, object_start) - 10:max(subject_end, object_end) + 10]

            relation_sentences.append({
                "sentence": relation_sentence,
                "subject": relation["subjectText"],
                "object": relation["objectText"],
                "relation": relation["rel_name"]
            })

    return entity_sentences, relation_sentences


def create_label_mappings(entity_sentences, relation_sentences):
    ner_labels = [item['entity'] for item in entity_sentences] + [None] * len(relation_sentences)
    re_labels = [None] * len(entity_sentences) + [item['relation'] for item in relation_sentences]

    ner_label2idx = {label: idx for idx, label in enumerate(set(ner_labels))}
    re_label2idx = {label: idx for idx, label in enumerate(set(re_labels))}
    ner_label2idx[None] = len(ner_label2idx) - 1
    re_label2idx[None] = len(re_label2idx) - 1

    idx2ner_label = {idx: label for label, idx in ner_label2idx.items()}
    idx2re_label = {idx: label for label, idx in re_label2idx.items()}

    return ner_label2idx, re_label2idx, idx2ner_label, idx2re_label
