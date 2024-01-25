import torch.optim as optim
import torch
from torch.utils.data import DataLoader, TensorDataset

from SentimentAnalysis.DatasetUtils import loadDataSet
from SentimentAnalysis.ResponseAnalysis import CustomerResponseAnalysis

if __name__ == '__main__':

    batch_size = 10
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    avg_loss = []
    avg_accuracy = []
    max_loop = 4

    model = CustomerResponseAnalysis(3)

    text, labels = loadDataSet("./data/train/train_VLSP.csv")
    train_input = torch.stack(text)
    train_labels = torch.stack(labels)

    train_dataset = TensorDataset(train_input, train_labels)

    dataLoader = DataLoader(dataset=train_dataset, shuffle=True, batch_size=batch_size)

    optimizer = optim.SGD(model.parameters(), lr=0.001, weight_decay=0.03)
    loss_fn = torch.nn.CrossEntropyLoss()

    model.train().cpu()

    for epoch in range(max_loop):
        for i, (inputs, labels) in enumerate(dataLoader):

            optimizer.zero_grad()

            output = model(inputs)
            loss = loss_fn(output, labels)

            output = output.view(10,3)
            _, label_actual = output.max(1)
            _, label_expect = labels.view(10,3).max(1)
            avg_loss.append(loss.item())
            loss = loss.mean()

            loss.backward(retain_graph=True)
            optimizer.step()
            print("epoch {}, step {}, actual correct {}/10%, loss {}"
                  .format(epoch,
                                  i,
                                  (label_actual == label_expect).sum().item(),
                                  loss.item()))


    # model.load_state_dict(torch.load("./models/Cnn.bin"))

    label_expected = torch.tensor([2])
    # avg_loss.append(torch.mean(torch.log(output, label_expected.view())))



