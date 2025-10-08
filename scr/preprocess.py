import pandas as pd
import numpy as np


from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

import torch
import torch.nn as nn
import torch.nn.functional as F

def create_dataset(df, window_size=30, overlap_step=10, load_threshold=10):
    """
    Create sliding-window datasets for time series classification.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing columns
    window_size : int
        Number of time steps in each window.
    overlap_step : int
        Step size for the sliding window.
    load_threshold : float
        Minimum sum of 'stress' labels in a window to label the window as stress (1), otherwise 0.

    Returns
    -------
    X : torch.Tensor
        Array of shape (num_windows, window_size, num_features) with input features.
    y : torch.Tensor
        Array of shape (num_windows,) with binary labels for each window.
    time : list
        List of time arrays corresponding to each window.
    """
    X, y, y_all, time = [], [], [], []
    for i in range(0, len(df) - window_size, overlap_step):
        X.append(df.iloc[i:i + window_size][['HR', 'EDA', 'TEMP', 'ACC_X', 'ACC_Y', 'ACC_Z', 'BVP']].values)
        y_all = df.iloc[i:i + window_size]['stress'].values
        time.append(df.iloc[i:i + window_size]['time'].values)

        # y.append(df.iloc[i + window_size - 1]['stress'])
        if sum(y_all) > load_threshold:
            y.append(1)
        else:
            y.append(0)
            
    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.float32)
    return torch.tensor(X), torch.tensor(y), time

def scaler_per_user(df):
    """
    Standardize the features for a single user using z-score normalization.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing columns ['HR', 'EDA', 'TEMP', 'ACC_X', 'ACC_Y', 'ACC_Z', 'BVP'].

    Returns
    -------
    df_scaled : pd.DataFrame
        DataFrame with standardized features.
    scaler : StandardScaler
        Fitted sklearn StandardScaler object.
    """
    scaler = StandardScaler()
    df[['HR', 'EDA', 'TEMP', 'ACC_X', 'ACC_Y', 'ACC_Z', 'BVP']] = scaler.fit_transform(df[['HR', 'EDA', 'TEMP', 'ACC_X', 'ACC_Y', 'ACC_Z', 'BVP']])
    return df, scaler

def trim_after_last_one(df, label_col='stress'):
    """
    Trim a DataFrame after the last occurrence of a label value 1.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    label_col : str
        Name of the column containing binary labels.

    Returns
    -------
    df_trimmed : pd.DataFrame
        DataFrame truncated after the last occurrence of 1. Empty if no 1 exists.
    """
    last_one_idx = df[df[label_col] == 1].index.max()

    if pd.notna(last_one_idx):
        df_trimmed = df.iloc[:last_one_idx + 1]
    else:
        df_trimmed = df.iloc[0:0]

    return df_trimmed

def plot_loss(loss_train, loss_test, title ='Loss'):
    """
    Plot training and testing loss (or other metric) over epochs.

    Parameters
    ----------
    loss_train : list or np.ndarray
        Training loss values per epoch.
    loss_test : list or np.ndarray
        Testing loss values per epoch.
    title : str
        Title of the plot (default 'Loss').
    """
    plt.figure(figsize=(10, 5))
    plt.plot(loss_train, label='Train')
    plt.plot(loss_test, label='Test')
    # plt.plot(f1_score_train, label='Train F1 Score')
    # plt.plot(f1_score_test, label='Test F1 Score')
    plt.xlabel('Epochs')
    plt.ylabel(title)
    plt.title(title + ' over Epochs')
    plt.legend()
    plt.show()


def plot_confusion_matrix(y_true, y_pred, title='Confusion Matrix'):
    """
    Plot a confusion matrix with color coding and labels.

    Parameters
    ----------
    y_true : array-like
        Ground truth binary labels.
    y_pred : array-like
        Predicted binary labels.
    title : str
        Title of the plot.
    """    
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(2)
    plt.xticks(tick_marks, ['No Stress', 'Stress'], rotation=45)
    plt.yticks(tick_marks, ['No Stress', 'Stress'])
    
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, cm[i, j], horizontalalignment='center',
                     color='white' if cm[i, j] > thresh else 'black')
    
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.show()    