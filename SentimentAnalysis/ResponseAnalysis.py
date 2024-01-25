from torch import nn
from transformers import AutoModel
import torch

class CustomerResponseAnalysis(nn.Module):

    def __init__(self, num_labels):
        super(CustomerResponseAnalysis, self).__init__()
        self.model = AutoModel.from_pretrained("vinai/phobert-base-v2", return_dict=False)
        self.num_labels = num_labels
        self.dropout = nn.Dropout(0.2)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(768, 512)
        self.fc2 = nn.Linear(512, self.num_labels)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, input_ids):
        y = self.extra_phoBert_output(input_ids)
        y = self.fc1(y)
        y = self.relu(y)
        y = self.dropout(y)

        y = self.fc2(y)
        y = self.softmax(y)
        return y

    def extra_phoBert_output(self, input_ids):
        y = []

        for input_id in input_ids:
            last_hidden_state, cls_hs = self.model(input_id)
            y.append(last_hidden_state[:, 0, :])

        return torch.stack(y)