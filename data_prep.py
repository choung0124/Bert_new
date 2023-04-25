import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer

class EntityRelationDataset(Dataset):
    def __init__(self, data, tokenizer, max_length):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data.iloc[idx]
        sentence = item["sentence"]
        subject_entity = item["subject_entity"]
        object_entity = item["object_entity"]
        relationship = item["relationship"]

        encoding = self.tokenizer.encode_plus(
            sentence,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'subject_entity': torch.tensor(subject_entity, dtype=torch.long),
            'object_entity': torch.tensor(object_entity, dtype=torch.long),
            'relationship': torch.tensor(relationship, dtype=torch.long)
        }

# Initialize the tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Create the datasets
train_dataset = EntityRelationDataset(train_data, tokenizer, max_length=128)
val_dataset = EntityRelationDataset(val_data, tokenizer, max_length=128)

# Create the data loaders
train_data_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_data_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
