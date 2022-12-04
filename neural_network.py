#%%

import torch.nn as nn
import torch
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error
import numpy as np
import time

import datetime 
from torch.utils.tensorboard import SummaryWriter

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


# Load dataset
X_train = np.load('data/array/X_train.npy')
X_test = np.load('data/array/X_test.npy')
y_train = np.load('data/array/y_train.npy')
y_test = np.load('data/array/y_test.npy')


class Dataset(torch.utils.data.Dataset):
  '''
  Prepare the Boston dataset for regression
  '''

  def __init__(self, X, y, scale_data=True):
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


class MLP(nn.Module):
  '''
    Multilayer Perceptron for regression.
  '''
  def __init__(self):
    super().__init__()
    self.layers = nn.Sequential(
      nn.Linear(125, 64),
      nn.ReLU(),
      nn.Linear(64, 32),
      nn.ReLU(),
      nn.Linear(32, 1),
      nn.ReLU(),
    )


  def forward(self, x):
    '''
      Forward pass
    '''
    return self.layers(x)




# Prepare dataset
dataset = Dataset(X_train, y_train, scale_data=False)
trainloader = torch.utils.data.DataLoader(dataset, batch_size=10, shuffle=True, num_workers=0)

mlp = MLP().to(device)

# Define the loss function and optimizer
loss_function = nn.MSELoss(reduction="mean")
optimizer = torch.optim.Adam(mlp.parameters(), lr=1e-4)

now = datetime.datetime.now()
writer_dir = "./logs/" + now.strftime('%m.%d/%H.%M') + '/'

tensorboard_writer = SummaryWriter(writer_dir)

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
        
        # Get and prepare inputs
        inputs, targets = data
        inputs, targets = inputs.float(), targets.float()
        targets = targets.reshape((targets.shape[0], 1))

        inputs = inputs.to(device)
        targets = targets.to(device)
        
        # Zero the gradients
        optimizer.zero_grad()
        
        # Perform forward pass
        mlp = mlp.to(device)
        outputs = mlp(inputs)

        output_ = (outputs.detach().cpu().numpy() > 0)

        y_train_pred.extend(output_)  # save prediction
        
        # Compute loss
        loss = loss_function(outputs, targets)

        target = target.data.cpu().numpy()
        y_train_true.extend(target)  # save ground truth
        
        # Perform backward pass
        loss.backward()
        
        # Perform optimization
        optimizer.step()
  

    elapsed_time = time.time() - start_time
    print("took {} seconds for fitting".format(elapsed_time))

# Process is complete.
print('Training process has finished.')

#%%

y_pred = mlp(torch.from_numpy(X_test).float()).detach().numpy()
y_pred = [int(value) if value >= 0 else 0 for value in y_pred]
print("Prediction error:", mean_absolute_error(y_true=y_test, y_pred=y_pred))
# %%