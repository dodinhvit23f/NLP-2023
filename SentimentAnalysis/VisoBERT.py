import numpy as np
import torch.optim as optim
import torch
from torch.utils.data import DataLoader

from SentimentAnalysis.DatasetUtils import CustomDataDataSet
from SentimentAnalysis.ResponseAnalysis import CustomerResponseAnalysis

if __name__ == '__main__':

    batch_size = 10

    avg_loss = []
    avg_accuracy = []

    model = CustomerResponseAnalysis(3)

    data_set = CustomDataDataSet("./data/train/train_VLSP.csv")
    dataLoader = DataLoader(dataset=data_set, shuffle=True, batch_size=batch_size)

    optimizer = optim.AdamW(model.parameters(), lr=0.001)
    loss_fn = torch.nn.CrossEntropyLoss()

    max_loop = 4

    model.train().cpu()

    for epoch in range(max_loop):
        for i, (inputs, labels) in enumerate(dataLoader):
            optimizer.step()
            optimizer.zero_grad()

            output = model(inputs)
            _, label_actual = output.max(2)

            loss = loss_fn(label_actual.type(torch.FloatTensor), labels)

            avg_loss.append(loss.item())
            loss = loss.mean()
            print(loss.item())
            loss.backward()

        optimizer.step()

    # model.load_state_dict(torch.load("./models/Cnn.bin"))

    label_expected = torch.tensor([2])
    # avg_loss.append(torch.mean(torch.log(output, label_expected.view())))



