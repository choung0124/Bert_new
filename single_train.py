import os
import json
import socket
import torch
import torch.distributed as dist
import argparse
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from BERT_train import train, BertForNERAndRE, config, num_ner_labels, num_re_labels, dataset, custom_collate_fn, num_workers, num_epochs, learning_rate, tokenizer, label_to_id, relation_to_id, NERRE_Dataset, preprocess_data, validate_json


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_gpus = torch.cuda.device_count()

    preprocessed_data = []

    json_directory = "test"

    for filename in os.listdir(json_directory):
        if filename.endswith(".json"):
            with open(os.path.join(json_directory, filename), "r") as f:
                json_data = json.load(f)
                if validate_json(json_data):
                    preprocessed_data.extend(preprocess_data(json_data, tokenizer, label_to_id, relation_to_id))

    # Create the dataset
    dataset = NERRE_Dataset(preprocessed_data, tokenizer, max_length=128, label_to_id=label_to_id, relation_to_id=relation_to_id)

    # Set the device and rank for the current process
    device = torch.device("cuda")

    # Replace the DataLoader with a DistributedSampler
    batch_size = 8  # Define the batch size
    dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=custom_collate_fn, num_workers=num_workers)

    # Initialize the model
    model = BertForNERAndRE(config, num_ner_labels, num_re_labels)
    model = model.to(device)

    # Use DataParallel for multi-GPU training if more than one GPU is available
    if num_gpus > 1:
        model = nn.DataParallel(model)

    # Call the train function from train.py
    accumulation_steps = 1  # Define the number of accumulation steps
    model = train(model, dataloader, device, num_epochs, learning_rate, accumulation_steps)

    # Save the model, tokenizer, and mappings
    output_dir = "models/combined"
    os.makedirs(output_dir, exist_ok=True)

    # Handle DataParallel model saving
    if isinstance(model, nn.DataParallel):
        model = model.module

    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    # Save the label_to_id and relation_to_id mappings
    with open(os.path.join(output_dir, "label_to_id.json"), "w") as f:
        json.dump(label_to_id, f)

    with open(os.path.join(output_dir, "relation_to_id.json"), "w") as f:
        json.dump(relation_to_id, f)

if __name__ == "__main__":
    main()
