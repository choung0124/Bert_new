import sys
import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizerFast
from sklearn.metrics import classification_report

from data_processing import read_json_files, extract_sentences, create_label_mappings
from dataset import NERRE_Dataset
from model import NER_RE_Model
from train import tokenize_our_data

def evaluate(model, dataloader, device):
    model.eval()
    ner_preds = []
    re_preds = []
    ner_true = []
    re_true = []

    with torch.no_grad():
        for batch in dataloader:
            tokens, ner_labels, re_labels = batch
            input_ids = tokens['input_ids'].to(device)
            attention_mask = tokens['attention_mask'].to(device)
            token_type_ids = tokens['token_type_ids'].to(device)

            ner_logits, re_logits = model(input_ids, attention_mask, token_type_ids)
            ner_pred = torch.argmax(ner_logits, dim=1).cpu().numpy()
            re_pred = torch.argmax(re_logits, dim=1).cpu().numpy()

            ner_preds.extend(ner_pred)
            re_preds.extend(re_pred)
            ner_true.extend(ner_labels.numpy())
            re_true.extend(re_labels.numpy())

    return ner_true, ner_preds, re_true, re_preds

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python evaluate.py <directory_name> <model_path>")
        sys.exit(1)

    dir_path = sys.argv[1]
    model_path = sys.argv[2]

    full_text, full_entities, full_relations = read_json_files(dir_path)
    entity_sentences, relation_sentences = extract_sentences(full_text, full_entities, full_relations)
    ner_label2idx, re_label2idx, idx2ner_label, idx2re_label = create_label_mappings(entity_sentences, relation_sentences)

    sentences = [item['sentence'] for item in entity_sentences + relation_sentences]
    ner_labels = [item['entity'] for item in entity_sentences] + [None] * len(relation_sentences)
    re_labels = [None] * len(entity_sentences) + [item['relation'] for item in relation_sentences]

    dataset = NERRE_Dataset(sentences, ner_labels, re_labels)
    tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
    tokenized_data = tokenize_our_data(dataset, tokenizer)
    
    # Load the saved model's state_dict and configuration
    checkpoint = torch.load(model_path)

    # Create the model with the saved configuration
    model = NER_RE_Model(ner_dim=checkpoint['ner_classifier_dim'],
                         re_dim=checkpoint['re_classifier_dim'])

    model.load_state_dict(checkpoint['model_state_dict'])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    dataloader = DataLoader(tokenized_data, batch_size=8, shuffle=False)
    ner_true, ner_preds, re_true, re_preds = evaluate(model, dataloader, device)

    ner_unique_labels = sorted(list(set(ner_true + ner_preds)))
    re_unique_labels = sorted(list(set(re_true + re_preds)))

    print("NER Evaluation:")
    print(classification_report(ner_true, ner_preds, labels=ner_unique_labels, target_names=[idx2ner_label[l] for l in ner_unique_labels], digits=4))

    print("RE Evaluation:")
    print(classification_report(re_true, re_preds, labels=re_unique_labels, target_names=[idx2re_label[l] for l in re_unique_labels], digits=4))

