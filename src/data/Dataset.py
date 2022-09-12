import numpy as np
from torch.utils.data import Dataset


class Dataset(Dataset):
    """
    Dataset holding training and testing data and distributing
    them in batches.
    """
    def __init__(self, x, y, tokenizer):
        self.labels = y
        self.texts = [tokenizer(text, return_tensors="pt", padding="max_length", max_length=512, truncation=True) for text in x]

    def classes(self):
        return self.labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        x = self.texts[idx]
        y = np.array(self.labels[idx])
        return x, y

