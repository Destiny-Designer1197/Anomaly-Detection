from __future__ import print_function, division
import os
import torch
import pandas as pd
from torch.utils.data import Dataset
from torchvision import transforms, utils
from sklearn.preprocessing import StandardScaler
from torch.nn.utils.rnn import pad_sequence
import pytorch_lightning as pl
from torch import nn
import torch.nn.functional as F
from sklearn.utils import resample
import warnings


warnings.filterwarnings("ignore")


dataset_directory = r"utils\dataset_files"


class TrainDataset(Dataset):
    """dataset."""

    def __init__(self, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        xy = pd.read_csv(os.path.join(dataset_directory, "Training_DS.csv"))
        xy.columns = [
            "Timestamp",
            "CAN_ID",
            "RTR",
            "DLC",
            "Data0",
            "Data1",
            "Data2",
            "Data3",
            "Data4",
            "Data5",
            "Data6",
            "Data7",
            "Mean",
            "Median",
            "Skew",
            "Kurtosis",
            "Variance",
            "Standard_deviation",
            "Label",
            "Anomaly_Label",
        ]
        # Downsampling the train dataset
        class_0 = xy[xy["Anomaly_Label"] == 0]
        class_1 = xy[xy["Anomaly_Label"] == 1]
        final_df = resample(
            class_0, replace=True, n_samples=len(class_1), random_state=42
        )
        down_sampled = pd.concat([final_df, class_1])
        #
        self.x = torch.from_numpy(
            down_sampled.loc[
                :,
                ~down_sampled.columns.isin(
                    [
                        "Label",
                        "Anomaly_Label",
                    ]
                ),
            ].to_numpy()
        )
        self.x = self.x.float()
        self.y = torch.from_numpy(down_sampled.loc[:, "Anomaly_Label"].to_numpy())
        self.y = self.y.long()
        self.n_samples = down_sampled.shape[0]
        self.transform = transform

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):

        return self.x[idx], self.y[idx]


class ValDataset(Dataset):
    """dataset."""

    def __init__(self, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        xy = pd.read_csv(os.path.join(dataset_directory, "Validation_DS.csv"))
        xy.columns = [
            "Timestamp",
            "CAN_ID",
            "RTR",
            "DLC",
            "Data0",
            "Data1",
            "Data2",
            "Data3",
            "Data4",
            "Data5",
            "Data6",
            "Data7",
            "Label",
            "Anomaly_Label",
            "Mean",
            "Median",
            "Skew",
            "Kurtosis",
            "Variance",
            "Standard_deviation",
        ]
        self.x = torch.from_numpy(
            xy.loc[
                :,
                ~xy.columns.isin(
                    [
                        "Label",
                        "Anomaly_Label",
                    ]
                ),
            ].to_numpy()
        )
        self.y = torch.from_numpy(xy.loc[:, "Anomaly_Label"].to_numpy())
        self.n_samples = xy.shape[0]
        self.x = self.x.float()
        self.y = self.y.long()
        self.transform = transform

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):

        return self.x[idx], self.y[idx]


class TestDataset(Dataset):
    """dataset."""

    def __init__(self, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        xy = pd.read_csv(os.path.join(dataset_directory, "Test_DS.csv"))
        xy.columns = [
            "Timestamp",
            "CAN_ID",
            "RTR",
            "DLC",
            "Data0",
            "Data1",
            "Data2",
            "Data3",
            "Data4",
            "Data5",
            "Data6",
            "Data7",
            "Label",
            "Anomaly_Label",
            "Mean",
            "Median",
            "Skew",
            "Kurtosis",
            "Variance",
            "Standard_deviation",
        ]
        self.x = torch.from_numpy(
            xy.loc[
                :,
                ~xy.columns.isin(
                    [
                        "Label",
                        "Anomaly_Label",
                    ]
                ),
            ].to_numpy()
        )
        self.x = self.x.float()
        self.y = torch.from_numpy(xy.loc[:, "Anomaly_Label"].to_numpy())
        self.y = self.y.long()
        self.n_samples = xy.shape[0]
        self.transform = transform

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):

        return self.x[idx], self.y[idx]
