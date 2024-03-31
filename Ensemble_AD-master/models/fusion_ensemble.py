import pytorch_lightning as pl
import torch
from torch import nn
import torch.nn.functional as F

# In this I have tried implementing Fusion Ensemble. Here I have combined two neural networks(Conv-LSTM and MLP) into one which gave 
# satisfying results.

class ConvLSTM(pl.LightningModule):

  def __init__(self):
    super().__init__()
    self.conv1 = nn.Conv1d(in_channels=1, out_channels=18, kernel_size=3)
    self.conv2 = nn.Conv1d(in_channels=18,out_channels=32, kernel_size=3)
    self.conv3 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3)
    self.conv4 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3)
    self.lstm1 = nn.LSTM(
        input_size= 10,
        hidden_size=128,
        num_layers=2,
    )
    self.fc = nn.Linear(128, 2)

  def forward(self, x):
    
    x = x.unsqueeze(1)
    x = F.relu(self.conv1(x))
    x = F.relu(self.conv2(x))
    x = F.relu(self.conv3(x))
    x = F.relu(self.conv4(x))
    x,_ = self.lstm1(x)
    x = x[:, -1, :]
    x = self.fc(x)
    return (x)

class MLP(pl.LightningModule):

    def __init__(self):
        super().__init__()
        self.linear1 = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=18, kernel_size=3,stride=1, padding=1),
            nn.MaxPool1d(3, stride=2),
            nn.ReLU())
        self.linear2 = nn.Linear(144, 128)
        self.fc = nn.Linear(128, 2)
        self.flatten = nn.Flatten()

    def forward(self, data):
        data = data.view(data.size(0), -1)
        # import pdb;
        # pdb.set_trace()
        output = F.relu(self.linear1(data.unsqueeze(1)))
        output = self.flatten(output)
        output = F.relu(self.linear2(output))
        output = self.fc(output)
        return output
        
class MyEnsemble(pl.LightningModule):
    def __init__(self, modelA, modelB, nb_classes=2):
        self.save_hyperparameters()
        super(MyEnsemble, self).__init__()
        self.modelA = modelA
        self.modelB = modelB
        # Remove last linear layer
        self.modelA.fc = nn.Identity()
        self.modelB.fc = nn.Identity()
        # Create new classifier
        self.classifier = nn.Linear(128+128, nb_classes)
        self.metric = torch.nn.CrossEntropyLoss()
        
    def forward(self, x):
        x1 = self.modelA(x.clone())  # clone to make sure x is not changed by inplace methods
        x1 = x1.view(x1.size(0), -1)
        x2 = self.modelB(x)
        x2 = x2.view(x2.size(0), -1)
        
        x = torch.cat((x1, x2), dim=1)
        
        x = self.classifier(F.relu(x))
        return x
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    def training_step(self, batch, batch_idx):
        x, y = batch
        loss = self.metric(self.forward(x), y)
        self.log("train_loss", loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        val_loss = self.metric(self.forward(x), y)
        self.log("val_loss", val_loss)
        return val_loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        test_loss = self.metric(self.forward(x), y)
        self.log("test_loss", test_loss)
        return test_loss    

