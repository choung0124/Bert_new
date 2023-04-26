import torch
from transformers import BertTokenizer, BertPreTrainedModel
from torch import nn

# Custom BERT model for NER and RE tasks
class BertForNERAndRE(BertPreTrainedModel):
    def __init__(self, config, num_ner_labels, num_re_labels):
        super().__init__(config)
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.ner_classifier = nn.Linear(config.hidden_size, num_ner_labels)
        self.re_classifier = nn.Linear(config.hidden_size, num_re_labels)

        self.num_ner_labels = num_ner_labels
        self.num_re_labels = num_re_labels

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        ner_labels=None,
        re_labels=None,
    ):
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )

        sequence_output = outputs[0]
        pooled_output = outputs[1]

        sequence_output = self.dropout(sequence_output)
        pooled_output = self.dropout(pooled_output)

        ner_logits = self.ner_classifier(sequence_output)
        re_logits = self.re_classifier(pooled_output)

        total_loss = 0
        if ner_labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            ner_loss = loss_fct(ner_logits.view(-1, self.config.num_ner_labels), ner_labels.view(-1))
            total_loss += ner_loss

        if re_labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            re_loss = loss_fct(re_logits.view(-1, self.config.num_re_labels), re_labels.view(-1))
            total_loss += re_loss

        return (total_loss, ner_logits, re_logits) if total_loss > 0 else (ner_logits, re_logits)

# Load the fine-tuned custom BERT model and tokenizer
model_dir = "models/combined"
tokenizer = BertTokenizer.from_pretrained(model_dir)
model = BertForNERAndRE.from_pretrained(model_dir, num_ner_labels=len(label_to_id), num_re_labels=len(relation_to_id))
model.eval()
model.to("cpu")

# Load the label_to_id and relation_to_id mappings
with open(os.path.join(model_dir, "label_to_id.json"), "r") as f:
    label_to_id = json.load(f)
id_to_label = {v: k for k, v in label_to_id.items()}

with open(os.path.join(model_dir, "relation_to_id.json"), "r") as f:
    relation_to_id = json.load(f)
id_to_relation = {v: k for k, v in relation_to_id.items()}

# Input text
input_text = "The incidence of myocardial injury following post-operative Goal Directed Therapy Background Studies suggest that Goal Directed Therapy (GDT) results in improved outcome following major surgery. However, there is concern that pre-emptive use of inotropic therapy may lead to an increased incidence of myocardial ischaemia and infarction. Methods Post hoc analysis of data collected prospectively during a randomised controlled trial of the effects of post-operative GDT in high-risk general surgical patients. Serum troponin T concentrations were measured at baseline and on"

# Tokenize the input text
encoded_input = tokenizer.encode_plus(input_text, return_tensors="pt", padding="max_length", truncation=True, max_length=512)

# Perform NER and RE tasks
with torch.no_grad():
    ner_logits, re_logits = model(**encoded_input)

# Decode NER predictions
ner_predictions = torch.argmax(ner_logits, dim=2).squeeze(0).tolist()
ner_labels = [id_to_label[pred] for pred in ner_predictions]

# Decode RE predictions
re_prediction = torch.argmax(re_logits, dim=1).item()
re_label = id_to_relation[re_prediction]

# Print results
print(f"NER labels: {ner_labels}")
print(f"Extracted relation: {re_label}")
