from datasets import load_dataset
import to_vector
# https://huggingface.co/datasets/aieng-lab/namexact/viewer/default/train
# dataset = load_dataset("aieng-lab/namexact", split="train")


# print(dataset)

# j = 0
# for i in dataset:
#     print(i)
#     j += 1
#     if j == 10:
#         break


from torch.utils.data import DataLoader, Dataset


# formatting the dataset



class NameDataset(Dataset):
    def __init__(self, split="train"):
        dataset = load_dataset("aieng-lab/namexact", split=split)
        formatted = []

        for point in dataset:
            name_tensor = to_vector.to_tensor(point["name"])
            if point["gender"] == 'M':
                gender = "Male"
            else:
                gender = "Female"
            label = to_vector.label_to_tensor(gender)

            formatted.append((name_tensor, label))

        self.data = formatted
    
    def __getitem__(self, index):
        return self.data[index]
    
    def __len__(self):
        return len(self.data)

nd = NameDataset()
dataloader = DataLoader(nd, batch_size=16, shuffle=True)

