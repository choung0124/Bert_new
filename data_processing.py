import json
import os
import re

def process_directory(dir_path):
    # Initialize lists to store the results
    full_text = []
    full_entities = []
    entity_sentences = []
    full_relations = []
    relation_sentences = []

    # Define a regular expression pattern to match sentence boundaries
    pattern = re.compile(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s')

    # Loop through each file in the directory
    for file_name in os.listdir(dir_path):
        if file_name.endswith('.json'):
            # Read the JSON file and extract the content field from each object
            with open(os.path.join(dir_path, file_name), 'r') as file:
                content = file.read()
                data = json.loads(content)
                text = data['text']
                entities = data['entities']
                relations = data['relation_info']
            full_text.append(text)
            full_entities.append(entities)
            full_relations.append(relations)

            # Find the sentence containing each entity
            for entity in entities:
                span_begin = entity['span']['begin']
                span_end = entity['span']['end']
                sentences = pattern.split(text)
                for sentence in sentences:
                    if text.find(sentence) <= span_begin and text.find(sentence) + len(sentence) >= span_end:
                        entity_sentences.append({'entity': entity['entityName'], 'sentence': sentence})

            # Find the sentence containing each relation
            for relation in relations:
                subject_text = relation['subjectText']
                object_text = relation['objectText']
                if subject_text in text and object_text in text:
                    sentences = pattern.split(text)
                    for sentence in sentences:
                        if subject_text in sentence and object_text in sentence:
                            relation_sentences.append({'relation': relation['rel_name'], 'sentence': sentence})
    return entity_sentences, relation_sentences
