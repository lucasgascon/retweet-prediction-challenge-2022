#%%

import torch.nn as nn
import torch
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error
import numpy as np
import time
import pandas as pd
import torch.nn as nn
from numpy import array
from torch.utils.data import TensorDataset
import random
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import datetime


# this ensures that the current MacOS version is at least 12.3+
print(torch.backends.mps.is_available())
# this ensures that the current current PyTorch installation was built with MPS activated.
print(torch.backends.mps.is_built())
if(torch.backends.mps.is_available() & torch.backends.mps.is_built()): 
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print('device : ', device)
device = 'cpu'

X_train = pd.read_csv('data/csv/X_train.csv').to_numpy()
X_test = pd.read_csv('data/csv/X_test.csv').to_numpy()
y_train = pd.read_csv('data/csv/y_train.csv', index_col=0).to_numpy()
y_test = pd.read_csv('data/csv/y_test.csv', index_col=0).to_numpy()



class Dataset(torch.utils.data.Dataset):
  '''
  Prepare the dataset for regression
  '''

  def __init__(self, X, y, scale_data=False):
    if not torch.is_tensor(X) and not torch.is_tensor(y):
      # Apply scaling if necessary
      if scale_data:
          X = StandardScaler().fit_transform(X)
      self.X = torch.from_numpy(X)
      self.y = torch.from_numpy(y)

  def __len__(self):
      return len(self.X)

  def __getitem__(self, i):
      return self.X[i], self.y[i]


class GenModel(nn.Module):
    """[LSTM Model Generator]

    """
    def __init__(self, hidden_dim,seq_length, n_layers,hidden_layers,
                 bidirectional, dropout=0.5):
        """[summary]

        Args:
            hidden_dim ([List]): [list of integers for dimensions of hidden layers]
            seq_length ([int]): [window size of 1 reading]
            n_layers ([int]): [description]
            hidden_layers ([int]): [description]
            bidirectional ([boolean]): [boolean of whether the bidirectional ]
            dropout (float, optional): [description]. Defaults to 0.5.
        """
        super().__init__()
        self.rnn = nn.LSTM(856, 
                           hidden_dim[0], 
                           num_layers=n_layers, #set to two: makes our LSTM 'deep'
                           bidirectional=bidirectional, #bidirectional or not
                           dropout=dropout,batch_first=True) #we add dropout for regularization
        
        if bidirectional:
            self.D=2
        else:
            self.D=1
        self.n_layers=n_layers
        self.hidden_dim=hidden_dim[0]
        self.nonlinearity = nn.ReLU() 
        self.hidden_layers = nn.ModuleList([])
        self.seq_length=seq_length
        self.dropout=nn.Dropout(dropout)
        assert(len(hidden_dim)>0)
        assert(len(hidden_dim)==1+hidden_layers)

        i=0
        if hidden_layers>0:
            self.hidden_layers.append(nn.Linear(hidden_dim[i]*self.D*self.seq_length, hidden_dim[i+1]))
            for i in range(hidden_layers-1):
                self.hidden_layers.append(nn.Linear(hidden_dim[i+1], hidden_dim[i+2]))
            self.output_projection = nn.Linear(hidden_dim[i+1], 1)
        else:
            self.output_projection = nn.Linear(hidden_dim[i]*self.D*self.seq_length, 1)
    
        
        
    def forward(self, x,hidden):
        """[Forward for Neural network]

        Args:
            x ([Tensor]): [input tensor for raw values]
            hidden ([Tensor]): [hidden state values for lstm model]

        Returns:
            [Tensor]: [output results from model]
        """
        
        batch_size= x.size(0)

        val, hidden = self.rnn(x,hidden) #feed to rnn
        
        #unpack sequence
        val = val.contiguous().view( batch_size,-1)
        for hidden_layer in self.hidden_layers:
              val = hidden_layer(val)
              val = self.dropout(val)
              val = self.nonlinearity(val) 
        out = self.output_projection(val)

        return out,hidden
    
    
    def init_hidden(self, batch_size):
        """[summary]

        Args:
            batch_size ([int]): [size of batch that you are inputting into the model]

        Returns:
            [Tensor]: [Returns a tensor with the dimensions equals to the dimensions of the model's
            hidden state with values 0]
        """
        weight = next(self.parameters()).data
        hidden = (weight.new(self.n_layers*self.D, batch_size, self.hidden_dim).zero_().to(device),
                        weight.new(self.n_layers*self.D, batch_size, self.hidden_dim).zero_().to(device))
        
        return hidden


batch_size = 10
model = GenModel([512], 30,2, 0, True,0.5)
model = model.to(device)
h = model.init_hidden(batch_size)

# Prepare dataset
train_dataset = Dataset(X_train, y_train, scale_data=False)
trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=10, shuffle=True, num_workers=0)

valid_dataset = Dataset(X_test, y_test, scale_data=False)
validloader = torch.utils.data.DataLoader(valid_dataset, batch_size=10, shuffle=True, num_workers=0)

# Define the loss function and optimizer
loss_function = nn.MSELoss(reduction="mean")
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# Run the training loop
for epoch in range(0, 50): # 5 epochs at maximum
    start_time = time.time()

    epoch_train_losses = []
    epoch_valid_losses = []

    # Print epoch
    print(f'Starting epoch {epoch+1}')

    # Set current loss value
    current_loss = 0.0

    y_train_pred = []
    y_train_true = []

    # Iterate over the DataLoader for training data
    for i, data in enumerate(trainloader, 0):

        h = model.init_hidden(batch_size)
        
        # Get and prepare inputs
        inputs, targets = data
        inputs, targets = inputs.float(), targets.float()
        targets = targets.reshape((targets.shape[0], 1))

        inputs = inputs.to(device)
        targets = targets.to(device)
        
        # Zero the gradients
        optimizer.zero_grad()
        
        # Perform forward pass
        model = model.to(device)
        outputs = model(inputs)

        outputs_ = outputs.detach().cpu().numpy()
        y_train_pred.extend(outputs_)  # save prediction
        
        # Compute loss
        loss = loss_function(outputs, targets)

        targets = targets.data.cpu().numpy()
        y_train_true.extend(targets)  # save ground truth
        
        # Perform backward pass
        loss.backward()
        
        # Perform optimization
        optimizer.step()

        epoch_train_losses.append(loss.detach().to('cpu'))


    y_valid_pred = []
    y_valid_true = []

    for i, data in enumerate(validloader, 0):

        # Get and prepare inputs
        inputs, targets = data
        inputs, targets = inputs.float(), targets.float()
        targets = targets.reshape((targets.shape[0], 1))

        inputs = inputs.to(device)
        targets = targets.to(device)

        model = model.to(device)
        outputs = model(inputs)

        outputs_ = outputs.detach().cpu().numpy()
        y_valid_pred.extend(outputs_)  # save prediction

        loss = loss_function(outputs, targets)

        targets = targets.data.cpu().numpy()
        y_valid_true.extend(targets)  # save ground truth

        epoch_valid_losses.append(loss.detach().cpu())
      
    
    # train_loss = sum(epoch_train_losses) / len(epoch_train_losses)
    # valid_loss = sum(epoch_valid_losses) / len(epoch_valid_losses)

    # tensorboard_writer.add_scalar(
    #     'Training epoch loss',
    #     train_loss,
    #     epoch)
    # tensorboard_writer.add_scalar(
    #     'Valid epoch loss',
    #     valid_loss,
    #     epoch)

    elapsed_time = time.time() - start_time
    print("took {} seconds for fitting".format(elapsed_time))

# Process is complete.
print('Training process has finished.')



#%%

y_pred = model(torch.from_numpy(X_test).float()).detach().numpy()
y_pred = [int(value) if value >= 0 else 0 for value in y_pred]
print("Prediction error:", mean_absolute_error(y_true=y_test, y_pred=y_pred))
# %%