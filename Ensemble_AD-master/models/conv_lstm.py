import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch import nn

class ConvLSTM(pl.LightningModule):

  def __init__(self):
    super().__init__()
    self.conv1 = torch.nn.Conv1d(in_channels=1, out_channels=18, kernel_size=3)
    self.conv2 = torch.nn.Conv1d(in_channels=18,out_channels=32, kernel_size=3)
    self.conv3 = torch.nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3)
    self.lstm1 = torch.nn.LSTM(
        input_size= 12,
        hidden_size=64,
        num_layers=2,
    )
    self.fc = torch.nn.Linear(64, 2)
    self.metric = torch.nn.CrossEntropyLoss()

  def forward(self, x):
    
    x = x.unsqueeze(1)
    x = F.relu(self.conv1(x))
    x = F.relu(self.conv2(x))
    x = F.relu(self.conv3(x))
    x,_ = self.lstm1(x)
    x = x[:, -1, :]
    x = self.fc(x)
    return (x)

  def configure_optimizers(self):

    optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
    return optimizer

  def training_step(self, batch):
    x, y = batch
    loss = self.metric(self.forward(x), y)
    self.log("train_loss", loss)
    return loss

