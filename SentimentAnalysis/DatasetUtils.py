import csv

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from torch.utils.data import Dataset, DataLoader


class CustomDataDataSet(Dataset):
    def __init__(self, path):
        self.tokenizer = AutoTokenizer.from_pretrained('uitnlp/visobert')
        self.model = AutoModel.from_pretrained('uitnlp/visobert')
        self.x = []
        self.y = []

        skip = False

        with open(path, newline='') as f:
            reader = csv.reader(f, delimiter='\t', quoting=csv.QUOTE_NONE)
            for row in reader:
                if skip:
                    x = self.tokenizer(row[1], return_tensors='pt')
                    self.x.append(self.model(**x)[1])
                    self.y.append([int(row[2])])
                skip = True

        self.number_of_sample = len(self.x)
        self.x = torch.stack(self.x)
        self.y = torch.FloatTensor(self.y)

    def __getitem__(self, item):
        return self.x[item], self.y[item]

    def __len__(self):
        return self.number_of_sample


if __name__ == '__main__':

    CustomDataDataSet("./data/train/train_VLSP.csv")
