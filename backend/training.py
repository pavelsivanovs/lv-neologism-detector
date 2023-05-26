import pandas as pd
import torch
from progiter import ProgIter
from torch import nn
from torch.utils import data
from torch.utils.data import random_split

from utils.logger import get_configured_logger

logger = get_configured_logger(__name__)


class NeologismClassificator(nn.Module):

    def __init__(self, input_size):
        super().__init__()
        self.lin_1 = nn.Linear(input_size, 16)
        self.lin_2 = nn.Linear(16, 8)
        self.lin_3 = nn.Linear(8, 1)
        self.act_fn = nn.Sigmoid()

    def forward(self, x):
        x = self.lin_1(x)
        x = self.act_fn(x)
        x = self.lin_2(x)
        x = self.act_fn(x)
        x = self.lin_3(x)
        x = self.act_fn(x)
        return x


class NeologismClassificatorDataset(data.Dataset):
    def __init__(self, corpus='pandemics'):
        if corpus == 'pandemics':
            features = pd.read_csv('dairies_features_onehot.csv', dtype=float)
            self.input_data = torch.tensor(features.values, dtype=torch.float32)
            result = pd.read_csv('dairies_result.csv', dtype=float)
            self.result = torch.tensor(result.values, dtype=torch.float32)
        elif corpus == 'commoncrawl':
            pass
        else:
            raise Exception('Unknown corpus')

    def __len__(self):
        return len(self.input_data)

    def __getitem__(self, idx):
        return self.input_data[idx], self.result[idx]


if __name__ == '__main__':
    logger.info(f'Availability of GPU: {torch.cuda.is_available()}')
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    logger.info(f'Device: {device}')

    epoch_count = 3000
    learning_rate = 0.01
    batch_size = 64

    # TODO set up plotting

    model = NeologismClassificator(input_size=21).to(device)
    dataset = NeologismClassificatorDataset()
    train_set, test_set = random_split(dataset, [.8, .2])
    train_loader = data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = data.DataLoader(test_set, batch_size=batch_size, shuffle=True)

    logger.info('Starting training')

    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)  # tweak around the LR

    accuracy_history, precision_history = [], []
    for epoch in ProgIter(range(epoch_count)):
        accuracy, precision, total = 0, 0, 0
        for idx, (data_input, data_target) in enumerate(train_loader):
            optimizer.zero_grad()

            result = model(data_input)
            loss = nn.functional.binary_cross_entropy_with_logits(result, data_target)

            loss.backward()
            optimizer.step()

            accuracy += 1 if torch.equal(result.round(), data_target) else 0

            total += 1
            if epoch % 100 == 0:
                print(f'Epoch: {epoch} [{idx}/{train_loader.batch_size}] | Accuracy: {accuracy/total:.2%}')

    # print(accuracy_history)
    # print(precision_history)

    # testing
    logger.info('Starting testing')
    with torch.no_grad():
        accuracy, precision, total = 0, 0, 0
        model.eval()
        for inputs, results in test_loader:
            outputs = model(inputs)
            accuracy += 1 if torch.equal(outputs.round(), results) else 0
            total += 1
        print(f'Accuracy: {accuracy/total:.2%}')
        print(f'Total: {total}')

    # model saving
    state_dict = model.state_dict()
    # print(state_dict)
    torch.save(state_dict, 'model.pt')

    # to load model
    # model = NeologismClassificator(input_size=21).to(device)
    # model.load_state_dict(torch.load('model.pt'))



