from torch.utils.data import Dataset
import torch
import random

class DETESTSDataset(Dataset):
    def __init__(self, data, add_context=True, task="stereotype", approach="hard"):
        self.data = data
        self.add_context = add_context
        self.task = task
        self.approach = approach
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data.loc[idx]

        source = sample["source"]
        id = sample["id"]
        comment_id = sample["comment_id"]

        text = sample["text"]
        context = ""

        if id.endswith("_a"):
            id = id.split("_a")[0]

        if self.add_context:
            if sample["source"] == "detests" and sample["level1"] != "0":
                idy = sample["level1"]
                context = random.choice(self.data[self.data["id"].str.startswith(idy)]["text"].values.tolist())
            if sample["source"] == "stereohoax" and sample["level3"] != "0":
                idy = sample["level3"]
                context = random.choice(self.data[self.data["id"].str.startswith(idy)]["text"].values.tolist())
        
        label = torch.tensor(sample[f"{self.task}_{self.approach}"])



        return {
            "idx": idx,
            "source": source,
            "id": sample["id"],
            "comment_id": comment_id,
            "text": (text, context),
            "label": label
        }


class TestDETESTSDataset(Dataset):
    def __init__(self, data):
        self.data = data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sample = self.data.loc[idx]

        id = sample["id"]

        text = sample["text"]
        context = ""

        if id.endswith("_a"):
            id = id.split("_a")[0]

        if sample["source"] == "detests" and sample["level1"] != "0":
            idy = sample["level1"]
            context = random.choice(self.data[self.data["id"].str.startswith(idy)]["text"].values.tolist())
        if sample["source"] == "stereohoax" and sample["level3"] != "0":
            try:
                idy = sample["level3"]
                context = random.choice(self.data[self.data["id"].str.startswith(idy)]["text"].values.tolist())
            except:
                context = ""
        
        return {
            "id": sample["id"],
            "text": (text, context),
        }
