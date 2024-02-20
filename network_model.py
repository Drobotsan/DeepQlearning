import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class DQNNet(nn.Module):

    def __init__(self, input_size, output_size, lr=1e-3, fc1_dims=400, fc2_dims=300):
        super(DQNNet, self).__init__()
        self.dense1 = nn.Linear(input_size, fc1_dims)
        self.dense2 = nn.Linear(fc1_dims, fc2_dims)
        self.dense3 = nn.Linear(fc2_dims, output_size)

        self.optimizer = optim.Adam(self.parameters(), lr=lr)

    def forward(self, x):
        x = F.relu(self.dense1(x))
        x = F.relu(self.dense2(x))
        x = self.dense3(x)
        return x

    def save_model(self, filename):

        torch.save(self.state_dict(), filename)

    def load_model(self, filename, device):

        # map_location is required to ensure that a model that is trained on GPU can be run even on CPU
        self.load_state_dict(torch.load(filename, map_location=device))