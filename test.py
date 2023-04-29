import torch
import logging
import pickle
import os
from transformers import DistilBertConfig, DistilBertTokenizerFast
from DistiliBERT_train import DistilBertForNERAndRE  # Assuming the model is defined in a separate file called 'model.py'

logging.getLogger("transformers").setLevel(logging.ERROR)

output_dir = "models/combined"
with open(os.path.join(output_dir, "label_to_id.pkl"), "rb") as f:
    label_to_id = pickle.load(f)

with open(os.path.join(output_dir, "relation_to_id.pkl"), "rb") as f:
    relation_to_id = pickle.load(f)

config = DistilBertConfig.from_pretrained("distilbert-base-uncased")
tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")

num_ner_labels = len(label_to_id)
num_re_labels = len(relation_to_id)
model = DistilBertForNERAndRE(config, num_ner_labels, num_re_labels)

model_path = "models/combined/pytorch_model.bin"  # Replace with the path to your trained model
model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
model.eval()

# Move the model to the appropriate device (GPU if available, otherwise CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
def extract_relationships(text, model, tokenizer, id_to_label, id_to_relation):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)

    # Move inputs to the same device as the model
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    # Run the model on the input text
    with torch.no_grad():
        outputs = model(**inputs)

    # Get the predicted NER and RE labels
    ner_predictions = torch.argmax(outputs["ner_logits"], dim=-1).squeeze().tolist()
    re_predictions = torch.argmax(outputs["re_logits"], dim=-1).squeeze().tolist()

    # Convert the predicted labels to their corresponding entity and relationship names
    ner_labels = [id_to_label[str(pred)] for pred in ner_predictions]
    re_labels = [[id_to_relation[str(pred)] for pred in row] for row in re_predictions]

    # Locate and print the subject and object entities along with their relationships
    for i, row in enumerate(re_labels):
        for j, relation in enumerate(row):
            if relation != "no_relation":  # Change this to the name of the "no relation" label in your dataset
                subject = ner_labels[i]
                object = ner_labels[j]
                print(f"{subject} entity: {object} entity: {relation}")

    return ner_labels, re_labels

input_text = "The incidence of myocardial injury following post-operative Goal Directed Therapy Background Studies suggest that Goal Directed Therapy (GDT) results in improved outcome following major surgery. However, there is concern that pre-emptive use of inotropic therapy may lead to an increased incidence of myocardial ischaemia and infarction. Methods Post hoc analysis of data collected prospectively during a randomised controlled trial of the effects of post-operative GDT in high-risk general surgical patients. Serum troponin T concentrations were measured at baseline and on day 1 and day 2 following surgery. Continuous ECG monitoring was performed during the eight hour intervention period. Patients were followed up for predefined cardiac complications. A univariate analysis was performed to identify any associations between potential risk factors for myocardial injury and elevated troponin T concentrations. Results GDT was associated with fewer complications, and a reduced duration of hospital stay. Troponin T concentrations above 0.01 mug l-1 were identified in eight patients in the GDT group and six in the control group. Values increased above 0.05 mug l-1 in four patients in the GDT group and two patients in the control group."
ner_labels, re_labels = extract_relationships(input_text, model, tokenizer, label_to_id, relation_to_id)
