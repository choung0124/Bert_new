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

    pattern = re.compile(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s')

    for text, entities, relations in zip(full_text, full_entities, full_relations):
        for entity in entities:
            span_begin = entity['span']['begin']
            span_end = entity['span']['end']
            sentences = pattern.split(text)
            for sentence in sentences:
                if text.find(sentence) <= span_begin and text.find(sentence) + len(sentence) >= span_end:
                    entity_sentences.append({'entity': entity['entityName'], 'sentence': sentence})

        for relation in relations:
            subject_text = relation['subjectText']
            object_text = relation['objectText']
            if subject_text in text and object_text in text:
                sentences = pattern.split(text)
                for sentence in sentences:
                    if subject_text in sentence and object_text in sentence:
                        relation_sentences.append({'relation': relation['rel_name'], 'sentence': sentence})

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
