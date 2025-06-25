
from datasets import load_dataset

# https://huggingface.co/datasets/aieng-lab/namexact/viewer/default/train
dataset = load_dataset("aieng-lab/namexact", split="train")

from torch.utils.data import Dataset, DataLoader
import to_vector
import torch

# j = 0
# for i in dataset:
#     print(i)
#     j += 1
#     if j > 10:
#         break

def process(name, label):
    return [to_vector.to_tensor(name), to_vector.label_to_tensor(label)]

def process_dataset(example):
    if example["gender"] == "M":
        gender = "Male"
    else:
        gender = "Female"
    
    return process(example["name"], gender)

class NameDataset(Dataset):
    def __init__(self, hf_dataset):
        self.data = hf_dataset

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = self.data[idx][0]
        y = self.data[idx][1]
        return x, y


# processed = dataset.map(process_dataset)
processed = [process_dataset(i) for i in dataset]

inp = []
labels = []
for i in processed:
    inp.append(i[0])
    labels.append(i[1])

good = (torch.stack(inp), torch.stack(labels))
# print(good[1].shape)
train_dataset = NameDataset(processed)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
