import os
import json
import socket
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from BERT_train import train, BertForNERAndRE, config, num_ner_labels, num_re_labels, dataset, custom_collate_fn, num_workers, num_epochs, learning_rate, tokenizer, label_to_id, relation_to_id, NERRE_Dataset, preprocess_data, validate_json

def get_unused_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0))
        return s.getsockname()[1]

os.environ['MASTER_ADDR'] = '192.168.100.60'
os.environ['MASTER_PORT'] = str(get_unused_port())
os.environ['RANK'] = '0'
os.environ['WORLD_SIZE'] = '2'
os.environ['LOCAL_RANK'] = '0'

# Rest of the script remains the same


# Load and preprocess data
# Load and preprocess data
preprocessed_data = []

json_directory = "test"  # Define the directory containing the JSON files

for filename in os.listdir(json_directory):
    if filename.endswith(".json"):
        with open(os.path.join(json_directory, filename), "r") as f:
            json_data = json.load(f)
            if validate_json(json_data):
                preprocessed_data.extend(preprocess_data(json_data, tokenizer, label_to_id, relation_to_id))



# Create the dataset
dataset = NERRE_Dataset(preprocessed_data, tokenizer, max_length=512, label_to_id=label_to_id, relation_to_id=relation_to_id)

# Initialize the distributed environment
torch.cuda.set_device(local_rank)
dist.init_process_group(backend='nccl')

# Set the device and rank for the current process
device = torch.device("cuda", local_rank)

# Replace the DataLoader with a DistributedSampler
sampler = DistributedSampler(dataset, num_replicas=int(os.environ['WORLD_SIZE']), rank=local_rank)
dataloader = DataLoader(dataset, batch_size=batch_size, sampler=sampler, collate_fn=custom_collate_fn, num_workers=num_workers)

# Wrap the model with DDP
model = BertForNERAndRE(config, num_ner_labels, num_re_labels)
model = model.to(device)
model = DDP(model, device_ids=[local_rank], output_device=local_rank)

# Call the train function from train.py within try-finally block
try:
    model = train(model, dataloader, device, num_epochs, learning_rate, accumulation_steps)
finally:
    dist.destroy_process_group()

# Save the model only in the main process
if dist.get_rank() == 0:
    output_dir = "models/combined"
    os.makedirs(output_dir, exist_ok=True)
    model.module.save_pretrained(output_dir)  # Use model.module to access the underlying model
    tokenizer.save_pretrained(output_dir)

    # Save the label_to_id and relation_to_id mappings
    with open(os.path.join(output_dir, "label_to_id.json"), "w") as f:
        json.dump(label_to_id, f)

    with open(os.path.join(output_dir, "relation_to_id.json"), "w") as f:
        json.dump(relation_to_id, f)
