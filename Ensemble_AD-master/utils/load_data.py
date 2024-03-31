
import pandas as pd
import warnings
warnings.filterwarnings('always')
warnings.filterwarnings('ignore')
from sklearn.utils import resample
import os



def load_data_train():
    dataset_directory = r'utils\dataset_files'
    df = pd.read_csv(os.path.join(dataset_directory,"Training_DS.csv"))
    df.columns= ['Timestamp', 'CAN_ID', 'RTR', 'DLC', 'Data0', 'Data1', 'Data2', 'Data3', 'Data4', 'Data5', 'Data6', 'Data7', 'Mean', 'Median',\
              'Skew', 'Kurtosis', 'Variance', 'Standard_deviation','Label','Anomaly_Label']
    class_0 = df[df["Anomaly_Label"] == 0]
    class_1 = df[df["Anomaly_Label"] == 1]
    fin_downsample = resample(class_0,replace=True,n_samples=len(class_1),random_state=42)
    data_downsampled = pd.concat([fin_downsample, class_1])
    X_train = data_downsampled.loc[:, ~data_downsampled.columns.isin(['Label','Anomaly_Label','Mean','Median','Skew','Kurtosis','Variance','Standard_deviation'])]

    y_train = data_downsampled['Anomaly_Label']

    X_train = X_train.to_numpy()
    y_train = y_train.to_numpy()

    return X_train,y_train

def load_data_test():
    dataset_directory = r'utils\dataset_files'
    df = pd.read_csv(os.path.join(dataset_directory,"Test_DS.csv"))
    df.columns= ['Timestamp', 'CAN_ID', 'RTR', 'DLC', 'Data0', 'Data1', 'Data2', 'Data3', 'Data4', 'Data5', 'Data6', 'Data7', 'Label','Anomaly_Label',\
        'Mean', 'Median','Skew', 'Kurtosis', 'Variance', 'Standard_deviation']


    # shuffled_10 = int((10/100) * shuffled)
    X_test = df.loc[:, ~df.columns.isin(['Label','Anomaly_Label','Mean','Median','Skew','Kurtosis','Variance','Standard_deviation'])]

    y_test = df['Anomaly_Label']

    X_test =  X_test.to_numpy()
    y_test = y_test.to_numpy()
    return X_test, y_test

def load_data_val():
    dataset_directory = r'utils\dataset_files'
    df = pd.read_csv(os.path.join(dataset_directory,"Validation_DS.csv"))
    df.columns= ['Timestamp', 'CAN_ID', 'RTR', 'DLC', 'Data0', 'Data1', 'Data2', 'Data3', 'Data4', 'Data5', 'Data6', 'Data7', 'Label','Anomaly_Label',\
        'Mean', 'Median','Skew', 'Kurtosis', 'Variance', 'Standard_deviation']


    # shuffled_10 = int((10/100) * shuffled)
    X_val = df.loc[:, ~df.columns.isin(['Label','Anomaly_Label','Mean','Median','Skew','Kurtosis','Variance','Standard_deviation'])]

    y_val = df['Anomaly_Label']

    X_val = X_val.to_numpy()
    y_val = y_val.to_numpy()
    return X_val, y_val