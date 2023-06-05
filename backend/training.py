import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from matplotlib.ticker import PercentFormatter
from torch import nn
from torch.nn.functional import binary_cross_entropy
from torch.utils import data
from torch.utils.data import random_split, WeightedRandomSampler
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


def draw_graph(x_label, y_label, data, title, filename):
    plt.rcParams['font.family'] = 'serif'
    fig, ax = plt.subplots()

    x_data = list(range(1, len(data[0]['data']) + 1))

    for entry in data:
        a = np.polyfit(np.log(x_data), entry['data'], 1)
        y_smooth = a[0] * np.log(x_data) + a[1]
        ax.scatter(x_data, entry['data'], s=1)
        ax.plot(x_data, y_smooth, lw=3, label=entry['label'])

    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
    ax.legend()
    ax.grid(True)
    fig.savefig(filename)
    fig.clf()


def train_model(model, train_loader, epoch_count, learning_rate, modelname):
    logger.info('Starting training')

    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)  # tweak around the LR

    accuracy_history = []
    loss_history = []
    precision_history = []
    recall_history = []
    f_score_history = []

    # optimizer.zero_grad()
    # scheduler = StepLR(optimizer, step_size=30, gamma=0.1)

    for epoch in range(epoch_count):
        for idx, (data_input, data_target) in enumerate(train_loader, start=1):
            optimizer.zero_grad()
            results = model(data_input)
            # output = binary_cross_entropy_with_logits(results, data_target)
            output = binary_cross_entropy(results, data_target)

            output.backward()
            optimizer.step()
            # scheduler.step()

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

    draw_graph(x_label='Trenēšanas partija', y_label='Metrika', title='Modeļa metrikas trenēšanas gaitā',
               filename='training_metrics.png', data=[
            {'label': 'Pareizība', 'data': accuracy_history},
            {'label': 'Precizitāte', 'data': precision_history},
            {'label': 'Pārklājums', 'data': recall_history},
            {'label': 'F-mērs', 'data': f_score_history},
            {'label': 'Zaudējums', 'data': loss_history},
        ])

    # model saving
    state_dict = model.state_dict()
    torch.save(state_dict, modelname)


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

    draw_graph(x_label='Testēšanas epohas partija', y_label='Metrika', title='Modeļa metrikas testēšanas gaitā',
               filename='testing_metrics.png', data=[
            {'label': 'Pareizība', 'data': accuracy_history},
            {'label': 'Precizitāte', 'data': precision_history},
            {'label': 'Pārklājums', 'data': recall_history},
            {'label': 'F-mērs', 'data': f_score_history}], )

    logger.info(f'Average accuracy: {np.average(accuracy_history):.2%}')
    logger.info(f'Average precision: {np.average(precision_history):.2%}')
    logger.info(f'Average recall: {np.average(recall_history):.2%}')
    logger.info(f'Average F-score: {np.average(f_score_history):.2%}')


def calculate_weights(train_set, num_of_classes):
    len_data = len(train_set)
    count_per_class = [0] * num_of_classes
    for r in train_set:
        count_per_class[int(r[1])] += 1
    weight_per_class = [.0] * num_of_classes
    for i in range(num_of_classes):
        weight_per_class[i] = float(len_data) / float(count_per_class[i])
    weights = [0] * len_data
    for idx, r in enumerate(train_set):
        weights[idx] = weight_per_class[int(r[1])]
    return weights


if __name__ == '__main__':
    logger.info(f'Availability of GPU: {torch.cuda.is_available()}')
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    logger.info(f'Device: {device}')

    epoch_count = 10
    learning_rate = 0.1
    batch_size = 32

    model = NeologismClassificator(input_size=21).to(device)
    dataset = NeologismClassificatorDataset()
    train_set, test_set = random_split(dataset, [.7, .3])

    weights = calculate_weights(train_set, 2)
    logger.debug(weights)
    sampler = WeightedRandomSampler(weights, len(weights))

    train_loader = data.DataLoader(train_set, batch_size=batch_size, sampler=sampler)
    test_loader = data.DataLoader(test_set, batch_size=5, shuffle=True)

    # train_model(model, train_loader, epoch_count, learning_rate, modelname='model_with_random_sampler.pt')
    test_model(model, test_loader, 'model_with_random_sampler.pt')
