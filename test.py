import sys
import torch
from transformers import BertTokenizerFast
from dataset import NERRE_Dataset
from model import NER_RE_Model

def tokenize_input_text(text, tokenizer):
    tokens = tokenizer(text, truncation=True, padding='max_length', max_length=128, return_tensors='pt')
    input_ids = tokens['input_ids'].squeeze()
    attention_mask = tokens['attention_mask'].squeeze()
    token_type_ids = tokens['token_type_ids'].squeeze()

    # Add batch dimension
    input_ids = input_ids.unsqueeze(0)
    attention_mask = attention_mask.unsqueeze(0)
    token_type_ids = token_type_ids.unsqueeze(0)

    return input_ids, attention_mask, token_type_ids

def predict(model, input_ids, attention_mask, token_type_ids, device):
    model.eval()
    with torch.no_grad():
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        token_type_ids = token_type_ids.to(device)

        ner_logits_subject, ner_logits_object, re_logits = model(input_ids, attention_mask, token_type_ids)
        ner_predictions_subject = torch.argmax(ner_logits_subject, dim=-1).squeeze().cpu()
        ner_predictions_object = torch.argmax(ner_logits_object, dim=-1).squeeze().cpu()
        re_predictions = torch.argmax(re_logits, dim=-1).squeeze().cpu()

    return ner_predictions_subject, ner_predictions_object, re_predictions

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python predict.py <text>")
        sys.exit(1)

    text = sys.argv[1]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the model and its configuration
    checkpoint = torch.load("trained_model.pt")
    model = NER_RE_Model(checkpoint['ner_classifier_dim'], checkpoint['re_classifier_dim'])
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    idx2ner_label = checkpoint['idx2ner_label']
    idx2re_label = checkpoint['idx2re_label']

    tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
    input_ids, attention_mask, token_type_ids = tokenize_input_text(text, tokenizer)

    ner_predictions_subject, ner_predictions_object, re_predictions = predict(model, input_ids, attention_mask, token_type_ids, device)

    # Convert the input_ids back to tokens
    tokens = tokenizer.convert_ids_to_tokens(input_ids.squeeze())[1:-1]  # Remove [CLS] and [SEP] tokens

    subject, object_, relationship = extract_entities_and_relationship(tokens, ner_predictions_subject, ner_predictions_object, re_predictions, idx2ner_label, idx2re_label)

    print("Subject:", subject)
    print("Object:", object_)
    print("Relationship:", relationship)

