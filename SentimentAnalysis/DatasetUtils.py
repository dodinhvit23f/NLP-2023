import csv

import torch
from transformers import AutoTokenizer, AutoModel
from torch.utils.data import Dataset, DataLoader
import py_vncorenlp

class CustomDataDataSet(Dataset):
    def __init__(self, path):
        self.tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base-v2")
        self.model = AutoModel.from_pretrained("vinai/phobert-base-v2")
        rdrsegmenter = py_vncorenlp.VnCoreNLP(annotators=["wseg"])
        self.x = []
        self.y = []

        skip = False

        with open(path, newline='') as f:
            reader = csv.reader(f, delimiter='\t', quoting=csv.QUOTE_NONE)
            for row in reader:
                if skip:
                    x = self.tokenizer.encode(' '.join(rdrsegmenter.word_segment(row[1])))
                    x = torch.tensor([x])
                    x = self.model(x)
                    self.x.append(x['pooler_output'])

                    y_true = torch.zeros(3, dtype=torch.float)
                    y_true[int(row[2])] = 1
                    y_true = y_true.reshape(-1,1).t()
                    self.y.append(y_true)
                skip = True

        self.number_of_sample = len(self.x)
        self.x = torch.stack(self.x)
        #self.y = torch.FloatTensor(self.y)

    def __getitem__(self, item):
        return self.x[item], self.y[item]

    def __len__(self):
        return self.number_of_sample


if __name__ == '__main__':

    CustomDataDataSet("./data/train/train_VLSP.csv")
