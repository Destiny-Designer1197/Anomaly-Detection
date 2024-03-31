import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

class CANintelliIDS(pl.LightningModule):
    def __init__(self):
        super().__init__()
        
        # CNN layer
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=7, kernel_size=5)
        self.pool1 = nn.MaxPool1d(kernel_size=2)
        
        # GRU layer with attention
        self.gru = nn.GRU(input_size=7, hidden_size=64)
        self.attention = nn.Linear(64, 1)
        
        # Fully connected layer
        self.fc = nn.Linear(64, 1)
    
    def forward(self, x):
        #import pdb;pdb.set_trace()
        x = x.unsqueeze(1)
        
        # CNN layer
        x = self.conv1(x)
        x = F.relu(x)
        x = self.pool1(x)
        
        # GRU layer with attention
        x, _ = self.gru(x)
        a = self.attention(x)
        a = F.softmax(a, dim=1)
        x = torch.sum(x * a, dim=1)
        
        # Fully connected layer
        x = self.fc(x)
        x = torch.sigmoid(x)
        
        return x
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        y = y.unsqueeze(1)
        y = y.float()
        loss = F.binary_cross_entropy(y_hat, y)
        self.log('train_loss', loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        y = y.unsqueeze(1)
        y= y.float()
        loss = F.binary_cross_entropy(y_hat, y)
        self.log('val_loss', loss)
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.0001)
