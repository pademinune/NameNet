from datasets import load_dataset

# https://huggingface.co/datasets/aieng-lab/namexact
# https://huggingface.co/datasets/aieng-lab/namextend

import torch

import models.v1 as v1
import models.v2 as v2
import models.r1 as r1


from torch.utils.data import DataLoader, Dataset




class NameDataset(Dataset):
    def __init__(self, split="train"):
        # dataset = load_dataset("aieng-lab/namexact", split=split)

        if split == "test":
            dataset = load_dataset("aieng-lab/namexact", split=split)
            formatted = []

            for point in dataset:
                name_tensor = r1.to_tensor(point["name"]) # type: ignore
                if point["gender"] == 'M': # type: ignore
                    # gender = "Male"
                    label = torch.tensor(0)
                else:
                    # gender = "Female"
                    label = torch.tensor(1)
                # label = to_vector.label_to_tensor(gender)

                formatted.append((name_tensor, label))

            self.data = formatted
        else:
            dataset = load_dataset("aieng-lab/namextend", split=split)
            formatted = []

            for point in dataset:
                name_tensor = v2.to_tensor(point["name"]) # type: ignore
                # if using large dataset, use primary_gender. Otherwise use 'gender'
                if point["primary_gender"] == 'M': # type: ignore
                    # gender = "Male"
                    label = torch.tensor(0)
                else:
                    # gender = "Female"
                    label = torch.tensor(1)
                # label = to_vector.label_to_tensor(gender)

                name_tensor = name_tensor.int()
                formatted.append((name_tensor, label))

            self.data = formatted
    
    def __getitem__(self, index):
        return self.data[index]
    
    def __len__(self):
        return len(self.data)

nd = NameDataset()
dataloader = DataLoader(nd, batch_size=2000, shuffle=True)

