import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch import nn


class CNN(pl.LightningModule):

    def __init__(self):
        super().__init__()
        self.conv1d_layer_1 = nn.Sequential(
        nn.Conv1d(in_channels=1, out_channels=12, kernel_size=3,stride=1, padding=1),
        nn.MaxPool1d(3, stride=2),
        nn.ReLU())
        self.conv1d_layer_2 = nn.Sequential(
        nn.Conv1d(in_channels=12, out_channels=48, kernel_size=3,stride=1, padding=1),
        nn.MaxPool1d(3, stride=2),
        nn.ReLU())
        self.linear3 = nn.Linear(144,32)
        self.fc = nn.Linear(32, 2)
        self.flatten = nn.Flatten()
        self.metric = torch.nn.CrossEntropyLoss()

    def forward(self, data):
        data = data.view(data.size(0), -1)
        output = F.relu(self.conv1d_layer_1(data.unsqueeze(1)))
        output = F.relu(self.conv1d_layer_2(output))
        output = self.flatten(output)
        output = F.relu(self.linear3(output))
        output = self.fc(output)
        return output

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    def training_step(self, batch, batch_idx):
        x, y = batch
        loss = self.metric(self.forward(x), y)
        self.log("train_loss", loss)
        return loss