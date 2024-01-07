from torch import nn


class CustomerResponseAnalysis(nn.Module):

    def __init__(self, num_labels):
        super(CustomerResponseAnalysis, self).__init__()
        self.num_labels = num_labels
        self.qa_outputs = nn.Linear(768, self.num_labels)

    def forward(self, input_ids):
        return self.qa_outputs(input_ids)
