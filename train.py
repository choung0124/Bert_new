import sys
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import BertTokenizerFast
from tqdm import tqdm

from data_processing import read_json_files, extract_sentences, create_label_mappings
from dataset import NERRE_Dataset
from model import NER_RE_Model

def tokenize_our_data(dataset, tokenizer, ner_label2idx, re_label2idx):
    tokenized_data = []
    for sentence, subject_label, object_label, re_label in dataset:
        tokens = tokenizer(sentence, truncation=True, padding='max_length', max_length=128, return_tensors='pt')
        input_ids = tokens['input_ids'].squeeze()
        attention_mask = tokens['attention_mask'].squeeze()
        token_type_ids = tokens['token_type_ids'].squeeze()
        subject_label_tensor = torch.tensor(ner_label2idx[subject_label] if subject_label is not None else ner_label2idx[None], dtype=torch.long)
        object_label_tensor = torch.tensor(ner_label2idx[object_label] if object_label is not None else ner_label2idx[None], dtype=torch.long)
        re_label_tensor = torch.tensor(re_label2idx[re_label] if re_label is not None else re_label2idx[None], dtype=torch.long)
        tokenized_data.append((input_ids, attention_mask, token_type_ids, subject_label_tensor, object_label_tensor, re_label_tensor))
    return tokenized_data

def train(model, dataloader, optimizer, device):
    model.train()
    total_loss = 0

    progress_bar = tqdm(dataloader, desc="Training", unit="batch")
    for input_ids, attention_mask, token_type_ids, subject_labels, object_labels, re_labels in progress_bar:
        optimizer.zero_grad()

        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        token_type_ids = token_type_ids.to(device)
        subject_labels = subject_labels.to(device)
        object_labels = object_labels.to(device)
        re_labels = re_labels.to(device)

        ner_logits_subject, ner_logits_object, re_logits = model(input_ids, attention_mask, token_type_ids)
        ner_loss_subject = criterion(ner_logits_subject, subject_labels)
        ner_loss_object = criterion(ner_logits_object, object_labels)
        re_loss = criterion(re_logits, re_labels)
        loss = ner_loss_subject + ner_loss_object + re_loss

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        progress_bar.set_postfix({"loss": loss.item()})

    return total_loss / len(dataloader)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python train.py <directory_name>")
        sys.exit(1)

    dir_path = sys.argv[1]
    full_text, full_entities, full_relations = read_json_files(dir_path)
    entity_sentences, relation_sentences = extract_sentences(full_text, full_entities, full_relations)
    ner_label2idx, re_label2idx, idx2ner_label, idx2re_label = create_label_mappings(entity_sentences, relation_sentences)

    sentences = [item['sentence'] for item in entity_sentences + relation_sentences]
    subject_labels = [None] * len(entity_sentences) + [item['subject'] for item in relation_sentences]
    object_labels = [None] * len(entity_sentences) + [item['object'] for item in relation_sentences]
    re_labels = [None] * len(entity_sentences) + [item['relation'] for item in relation_sentences]

    dataset = NERRE_Dataset(sentences, subject_labels, object_labels, re_labels)

    tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
    tokenized_data = tokenize_our_data(dataset, tokenizer, ner_label2idx, re_label2idx)

    model = NER_RE_Model(len(ner_label2idx), len(re_label2idx))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    dataloader = DataLoader(tokenized_data, batch_size=8, shuffle=True)
    optimizer = AdamW(model.parameters(), lr=3e-5)
    criterion = torch.nn.CrossEntropyLoss()


    num_epochs = 3
    for epoch in range(num_epochs):
        print(f"{'-' * 15} Epoch {epoch + 1}/{num_epochs} {'-' * 15}")

        # Train the model
        avg_train_loss = train(model, dataloader, optimizer, device)
        print(f"{'*' * 10} Average Training Loss: {avg_train_loss:.4f} {'*' * 10}")

    # Save the model's state_dict and configuration
    torch.save({
        'model_state_dict': model.state_dict(),
        'ner_classifier_dim': model.ner_classifier.out_features,
        're_classifier_dim': model.re_classifier.out_features,
        'idx2ner_label': idx2ner_label,
        'idx2re_label': idx2re_label
    }, "trained_model.pt")



