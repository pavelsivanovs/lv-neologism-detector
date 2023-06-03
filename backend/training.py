import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from matplotlib.ticker import PercentFormatter
from torch import nn
from torch.nn.functional import binary_cross_entropy_with_logits, binary_cross_entropy
from torch.optim.lr_scheduler import StepLR
from torch.utils import data
from torch.utils.data import random_split
from torcheval.metrics.functional import binary_accuracy, binary_recall, binary_precision, binary_f1_score

from utils.logger import get_configured_logger

logger = get_configured_logger(__name__)


class NeologismClassificator(nn.Module):

    def __init__(self, input_size):
        super().__init__()
        self.lin_1 = nn.Linear(input_size, 64)
        self.lin_2 = nn.Linear(64, 32)
        self.lin_3 = nn.Linear(32, 16)
        self.lin_4 = nn.Linear(16, 1)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()
        self.norm1 = nn.BatchNorm1d(64)
        self.norm2 = nn.BatchNorm1d(32)
        self.norm3 = nn.BatchNorm1d(16)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.lin_1(x)
        x = self.norm1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.lin_2(x)
        x = self.norm2(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.lin_3(x)
        x = self.norm3(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.lin_4(x)
        x = self.sigmoid(x)
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
    precision_history = []
    recall_history = []
    f_score_history = []

    optimizer.zero_grad()
    scheduler = StepLR(optimizer, step_size=30, gamma=0.1)

    for epoch in range(epoch_count):
        for idx, (data_input, data_target) in enumerate(train_loader, start=1):
            results = model(data_input)
            # output = binary_cross_entropy_with_logits(results, data_target)
            output = binary_cross_entropy(results, data_target)

            output.backward()
            # optimizer.step()
            scheduler.step()

            results = results.round().squeeze().type(torch.int32)
            data_target = data_target.squeeze().type(torch.int32)

            batch_accuracy = binary_accuracy(results, data_target).item()
            batch_loss = output.item()
            precision = binary_precision(results, data_target).item()
            recall = binary_recall(results, data_target).item()
            f_score = binary_f1_score(results, data_target).item()

            accuracy_history.append(batch_accuracy)
            loss_history.append(batch_loss)
            precision_history.append(precision)
            recall_history.append(recall)
            f_score_history.append(f_score)

    batches = list(range(1, len(accuracy_history) + 1))
    a = np.polyfit(np.log(batches), accuracy_history, 1)
    accuracy_smooth = a[0] * np.log(batches) + a[1]
    a = np.polyfit(np.log(batches), loss_history, 1)
    loss_smooth = a[0] * np.log(batches) + a[1]
    a = np.polyfit(np.log(batches), precision_history, 1)
    precision_smooth = a[0] * np.log(batches) + a[1]
    a = np.polyfit(np.log(batches), recall_history, 1)
    recall_smooth = a[0] * np.log(batches) + a[1]
    a = np.polyfit(np.log(batches), f_score_history, 1)
    f_score_smooth = a[0] * np.log(batches) + a[1]

    plt.rcParams['font.family'] = 'serif'
    fig, ax = plt.subplots()

    ax.plot(batches, accuracy_smooth, lw=3, label='Pareizība')
    ax.scatter(batches, accuracy_history, s=1)
    ax.plot(batches, loss_smooth, lw=3, label='Zaudējums')
    ax.scatter(batches, loss_history, s=1)
    ax.scatter(batches, precision_history, s=1)
    ax.plot(batches, precision_smooth, lw=3, label='Precizitāte')
    ax.scatter(batches, recall_history, s=1)
    ax.plot(batches, recall_smooth, lw=3, label='Pārklājums')
    ax.scatter(batches, f_score_history, s=1)
    ax.plot(batches, f_score_smooth, lw=3, label='F-mērs')

    ax.set_title('Modeļa metrikas trenēšanas gaitā')
    ax.set_xlabel('Trenēšanas partija')
    ax.set_ylabel('Metrika')
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
    ax.legend()
    ax.grid(True)
    fig.savefig('training_metrics.png')
    fig.clf()

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

    with torch.no_grad(), open('test_results.csv', 'w', encoding='utf8'):
        model.eval()
        for idx, (inputs, targets) in enumerate(test_loader, start=1):
            results = model(inputs)

            logger.debug(results.squeeze())
            logger.debug(targets.squeeze())

            results = results.round().squeeze().type(torch.int32)
            targets = targets.squeeze().type(torch.int32)

            accuracy = binary_accuracy(results, targets).item()
            precision = binary_precision(results, targets).item()
            recall = binary_recall(results, targets).item()
            f_score = binary_f1_score(results, targets).item()

            accuracy_history.append(accuracy)
            precision_history.append(precision)
            recall_history.append(recall)
            f_score_history.append(f_score)

    batches = list(range(1, len(accuracy_history) + 1))
    a = np.polyfit(np.log(batches), accuracy_history, 1)
    accuracy_smooth = a[0] * np.log(batches) + a[1]
    a = np.polyfit(np.log(batches), precision_history, 1)
    precision_smooth = a[0] * np.log(batches) + a[1]
    a = np.polyfit(np.log(batches), recall_history, 1)
    recall_smooth = a[0] * np.log(batches) + a[1]
    a = np.polyfit(np.log(batches), f_score_history, 1)
    f_score_smooth = a[0] * np.log(batches) + a[1]

    plt.rcParams['font.family'] = 'serif'
    fig, ax = plt.subplots()

    ax.scatter(batches, accuracy_history, s=1)
    ax.plot(batches, accuracy_smooth, lw=3, label='Pareizība')
    ax.scatter(batches, precision_history, s=1)
    ax.plot(batches, precision_smooth, lw=3, label='Precizitāte')
    ax.scatter(batches, recall_history, s=1)
    ax.plot(batches, recall_smooth, lw=3, label='Pārklājums')
    ax.scatter(batches, f_score_history, s=1)
    ax.plot(batches, f_score_smooth, lw=3, label='F-mērs')

    ax.set_title('Modeļa metrikas testēšanas gaitā')
    ax.set_xlabel('Testēšanas epohas partija')
    ax.set_ylabel('Metrika')
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
    ax.legend()
    ax.grid()
    fig.savefig('testing_metrics.png')
    fig.clf()

    logger.info(f'Average accuracy: {np.average(accuracy_history):.2%}')
    logger.info(f'Average precision: {np.average(precision_history):.2%}')
    logger.info(f'Average recall: {np.average(recall_history):.2%}')
    logger.info(f'Average F-score: {np.average(f_score_history):.2%}')


if __name__ == '__main__':
    logger.info(f'Availability of GPU: {torch.cuda.is_available()}')
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    logger.info(f'Device: {device}')

    epoch_count = 20
    learning_rate = 0.1
    batch_size = 64

    model = NeologismClassificator(input_size=21).to(device)
    dataset = NeologismClassificatorDataset()
    train_set, test_set = random_split(dataset, [.5, .5])
    train_loader = data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = data.DataLoader(test_set, batch_size=5, shuffle=True)

    # train_model(model, train_loader, epoch_count, learning_rate)
    test_model(model, test_loader, 'model.pt')
