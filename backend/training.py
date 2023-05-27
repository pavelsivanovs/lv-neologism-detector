import matplotlib.pyplot as plt
import pandas as pd
import torch
from torch import nn
from torch.nn.functional import binary_cross_entropy_with_logits
from torch.utils import data
from torch.utils.data import random_split
from torcheval.metrics.functional import binary_accuracy, binary_recall, binary_precision, binary_f1_score

from utils.logger import get_configured_logger

logger = get_configured_logger(__name__)


class NeologismClassificator(nn.Module):

    def __init__(self, input_size):
        super().__init__()
        self.lin_1 = nn.Linear(input_size, 32)
        self.lin_2 = nn.Linear(32, 16)
        self.lin_3 = nn.Linear(16, 8)
        self.lin_4 = nn.Linear(8, 1)
        self.act_fn = nn.Sigmoid()

    def forward(self, x):
        x = self.lin_1(x)
        x = self.act_fn(x)
        x = self.lin_2(x)
        x = self.act_fn(x)
        x = self.lin_3(x)
        x = self.act_fn(x)
        x = self.lin_4(x)
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


def train_model(model, train_loader, epoch_count, learning_rate):
    logger.info('Starting training')

    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)  # tweak around the LR

    accuracy_history = []
    loss_history = []
    for epoch in range(epoch_count):
        for idx, (data_input, data_target) in enumerate(train_loader, start=1):
            optimizer.zero_grad()
            results = model(data_input)
            loss = binary_cross_entropy_with_logits(results, data_target)

            loss.backward()
            optimizer.step()

            results = results.round().squeeze().type(torch.int32)
            data_target = data_target.squeeze().type(torch.int32)

            batch_accuracy = binary_accuracy(results, data_target).item()
            batch_loss = loss.item()

            accuracy_history.append(batch_accuracy)
            loss_history.append(batch_loss)

    batches = list(range(1, len(accuracy_history) + 1))
    plt.rcParams['font.family'] = 'serif'
    plt.plot(batches, accuracy_history, label='Pareizība')
    plt.plot(batches, loss_history, label='Zaudējums')
    plt.xlabel('Trenēšanas epohas partija')
    plt.ylabel('Metrika')
    plt.title('Modeļa metrikas trenēšanas gaitā')
    plt.legend()
    plt.grid(True)
    plt.savefig('training_metrics.png')
    plt.clf()

    # model saving
    state_dict = model.state_dict()
    torch.save(state_dict, 'model.pt')


def test_model(model, test_loader, filename):
    logger.info('Starting testing')
    model.load_state_dict(torch.load(filename))

    accuracy_history = []
    precision_history = []
    recall_history = []
    f_score_history = []

    with torch.no_grad(), open('test_results.csv', 'w', encoding='utf8') as results:
        model.eval()
        for idx, (inputs, targets) in enumerate(test_loader, start=1):
            results = model(inputs)

            logger.debug(results.squeeze())
            logger.debug(targets.squeeze())

            results = results.round().squeeze().type(torch.int32)
            targets = targets.squeeze().type(torch.int32)

            accuracy = binary_accuracy(results, targets)
            precision = binary_precision(results, targets)
            recall = binary_recall(results, targets)
            f_score = binary_f1_score(results, targets)

            accuracy_history.append(accuracy)
            precision_history.append(precision)
            recall_history.append(recall)
            f_score_history.append(f_score)

    batches = list(range(1, len(accuracy_history) + 1))
    plt.rcParams['font.family'] = 'serif'
    plt.plot(batches, accuracy_history, label='Pareizība')
    plt.plot(batches, precision_history, label='Precizitāte')
    plt.plot(batches, recall_history, label='Pārklājums')
    plt.plot(batches, f_score_history, label='F-mērs')
    plt.xlabel('Testēšanas epohas partija')
    plt.ylabel('Metrika')
    plt.title('Modeļa metrikas testēšanas gaitā')
    plt.legend()
    plt.grid(True)
    plt.savefig('testing_metrics.png')
    plt.clf()


if __name__ == '__main__':
    logger.info(f'Availability of GPU: {torch.cuda.is_available()}')
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    logger.info(f'Device: {device}')

    epoch_count = 1
    learning_rate = 0.1
    batch_size = 200

    model = NeologismClassificator(input_size=21).to(device)
    dataset = NeologismClassificatorDataset()
    train_set, test_set = random_split(dataset, [.7, .3])
    train_loader = data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = data.DataLoader(test_set, batch_size=batch_size, shuffle=True)

    train_model(model, train_loader, epoch_count, learning_rate)
    test_model(model, test_loader, 'model.pt')
