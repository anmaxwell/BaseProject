import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self, input_shape, n_actions):
        super(Model, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(input_shape[0]*input_shape[1], 64),
            nn.ReLU(),
            nn.Linear(64, 128) 
        )
        
        self.actor = nn.Sequential(
            nn.Linear(128, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions)
        )

        self.critic = nn.Sequential(
            nn.Linear(128, 512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )


    def forward(self, x):
        # flatten the observation space Box to linear tensor
        x_flat = torch.flatten(x, 1,2).to(torch.float32)
        #print('x_flat', x_flat.size(), x_flat)
        init_out = self.net(x_flat)
        return self.actor(init_out), self.critic(init_out)