import torch
import torch.nn as nn

#create model
class Model(nn.Module):
    def __init__(self, input_shape, n_actions):
        super(Model, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(input_shape[0]*input_shape[1], 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, n_actions)
            
        )

    def forward(self, x):
        # flatten the observation space Box to linear tensor
        tensor_array = torch.from_numpy(x)
        x_flat = torch.flatten(tensor_array).to(torch.float32)
        return self.net(x_flat)