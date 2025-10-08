
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import torch
import torch.nn as nn
import torch.nn.functional as F



class SignalAttentionFusion(nn.Module):
    """
    SignalAttentionFusion module applies attention across signal channels.
    
    Parameters:
    -----------
    input_channels : int
        Number of input features per time step.
    attention_dim : int
        Dimension of the attention space.
    
    Attributes:
    -----------
    query, key, value : nn.Linear
        Linear layers to compute Q, K, V for attention.
    softmax : nn.Softmax
        Softmax to compute attention weights.
    latest_attn_weights : torch.Tensor
        Stores the last attention weights for analysis.
    """

    def __init__(self, input_channels=7, attention_dim=32):
        super(SignalAttentionFusion, self).__init__()
        self.query = nn.Linear(input_channels, attention_dim)
        self.key = nn.Linear(input_channels, attention_dim)
        self.value = nn.Linear(input_channels, attention_dim)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        # Apply attention across the 'channels' dimension per time step
        Q = self.query(x) 
        K = self.key(x)   
        V = self.value(x) 

        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / (K.size(-1) ** 0.5)
        attn_weights = self.softmax(attn_scores) 

        self.latest_attn_weights = attn_weights.detach().cpu()

        fused = torch.matmul(attn_weights, V)
        return fused



class StressClassifierWithFusion(nn.Module):
    """
    StressClassifierWithFusion applies attention fusion followed by an LSTM and classifier.
    
    Parameters:
    -----------
    input_channels : int
        Number of input features per time step.
    attention_dim : int
        Dimension of the attention layer.
    lstm_hidden : int
        Hidden size for the LSTM layer.
    
    Attributes:
    -----------
    attn_fusion : SignalAttentionFusion
        Attention module applied to input signals.
    lstm : nn.LSTM
        LSTM layer for temporal modeling.
    classifier : nn.Sequential
        Fully connected layers with Sigmoid for binary classification.
    """
    def __init__(self, input_channels=7, attention_dim=32, lstm_hidden=64):
        super(StressClassifierWithFusion, self).__init__()
        self.attn_fusion = SignalAttentionFusion(input_channels, attention_dim)
        self.lstm = nn.LSTM(attention_dim, lstm_hidden, batch_first=True)
        self.classifier = nn.Sequential(
            nn.Linear(lstm_hidden, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x_fused = self.attn_fusion(x)
        lstm_out, _ = self.lstm(x_fused) 
        out = lstm_out[:, -1, :] 
        return self.classifier(out).squeeze(-1)
    
    def get_attention_weights(self):
        return self.attn_fusion.latest_attn_weights


class TCN_net(nn.Module):
    """
    Temporal Convolutional Network (TCN) for time series classification.
    
    Parameters:
    -----------
    input_size : int
        Number of input features per time step.
    num_channels : list of int
        Number of channels for each TCN layer.
    kernel_size : int
        Kernel size for convolutions.
    output_size : int
        Number of output features (usually 1 for binary classification).
    
    Attributes:
    -----------
    tcn : nn.Sequential
        TCN layers with Conv1d, ReLU, Dropout, and BatchNorm.
    fc : nn.Linear
        Fully connected layer for final output.
    sigmoid : nn.Sigmoid
        Sigmoid activation binary output.
    """
    def __init__(self, input_size=7, num_channels=[64, 64], kernel_size=3, output_size=1):
        super(TCN_net, self).__init__()
        layers = []
        for i in range(len(num_channels)):
            layers.append(nn.Conv1d(input_size if i == 0 else num_channels[i-1], num_channels[i], kernel_size, padding=(kernel_size - 1) // 2))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.5))
            layers.append(nn.BatchNorm1d(num_channels[i]))

        self.tcn = nn.Sequential(*layers)
        self.fc = nn.Linear(num_channels[-1], output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = x.transpose(1, 2)  
        out = self.tcn(x)
        out = out[:, :, -1]
        out = self.fc(out)
        out = self.sigmoid(out)
        return out.squeeze()
    
class SimpleLSTM(nn.Module):
    """
    Simple LSTM network for sequence classification.
    
    Parameters:
    -----------
    input_size : int
        Number of input features per time step.
    hidden_size : int
        Hidden size of LSTM.
    num_layers : int
        Number of LSTM layers.
    output_size : int
        Number of outputs (usually 1 for binary classification).
    
    Attributes:
    -----------
    lstm : nn.LSTM
        LSTM layer(s) for temporal modeling.
    fc : nn.Linear
        Fully connected output layer.
    sigmoid : nn.Sigmoid
        Sigmoid activation for binary output.
    """
    def __init__(self, input_size=7, hidden_size=64, num_layers=2, output_size=1):
        super(SimpleLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.8, )
        # self.batch_norm = nn.BatchNorm1d(hidden_size)
        self.fc = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        out = self.fc(lstm_out[:, -1, :])
        # out = self.batch_norm(out)
        out = self.sigmoid(out)
        return out.squeeze()