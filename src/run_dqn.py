import numpy as np
import random
from collections import deque
from tensorboardX import SummaryWriter

import torch
import torch.nn.functional as F
import torch.optim as optim

from environments.env import SchedulerEnv
from models.dqn import Model

# Hyperparameters
batch_size = 32

gamma = 0.99

eps_start=1.0
eps_decay = 0.995
eps_min = 0.1      # Minimal exploration rate (epsilon-greedy)

num_rounds = 1500
#num_episodes = 500
learning_limit = 100
replay_limit = 1000  # Number of steps until starting replay
#weight_update = 1000 # Number of steps until updating the target weights


if __name__ == "__main__":

    #device = "cuda"
    device = "cpu"

    env = SchedulerEnv()

    #start writing to tensorboard
    writer = SummaryWriter(comment="Scheduler DQN")

    #create the current network and target network
    policy_model = Model(env.observation_space.shape, env.action_space.n).to(device)

    target_model = Model(env.observation_space.shape, env.action_space.n).to(device)
    target_model.load_state_dict(policy_model.state_dict())

    optimizer = optim.Adam(policy_model.parameters(), lr=0.001, eps=1e-3)

    # Exploration rate    
    replay_buffer = deque(maxlen=1000)

    step_idx = 0
    epsilon = eps_start

    for i in range(num_rounds):
        #change this for while not true once it works
        state = env.reset()
        episode_reward = 0
        done = False
        #print('reset here')

        for j in range(50):
    #    while not done:
            
            step_idx += 1
            #print(i,j,step_idx)


            #epsilon for epsilon greedy strategy  
            if epsilon > eps_min:
                epsilon *= eps_decay
                
            #print('epsilon', epsilon)   
            #check = policy_model(state)
            #print(check)

            # Select and perform an action
            if step_idx > learning_limit:
                if np.random.rand() > epsilon:
                    action = torch.argmax(policy_model(state))
            else:
                action = np.random.randint(env.action_space.n)

            next_state, reward, done, _ = env.step(action)
            reward = torch.tensor([reward], device=device)
            episode_reward += reward
            #print('here rewards', episode_reward, reward, step_idx)

            # Store other info in replay buffer
            replay_buffer.append((state, action, reward, next_state, done))
            #print('len buffer', len(replay_buffer))

            # Move to the next state
            state = next_state

            if done:
                break
                
        #print('stopped episode', j, episode_reward)

        writer.add_scalar('episode_reward', episode_reward, i)
            
        #print('step', step_idx, 'i', i, 'j', j, episode_reward)

        #once we're ready to learn then start learning with mini batches
        if len(replay_buffer) == replay_limit:
            #print('replay buffer')
            optimizer.zero_grad()
            
            minibatch = random.sample(replay_buffer, batch_size)

            for state, action, reward, next_state, done in minibatch:    
                #pass state to policy to get qval from policy
                pred_qval = max(policy_model(state)) 

            #pass next state to target policy to get next set of qvals (future gains)
            if not done:
                next_qval = (reward + (gamma * max(target_model(next_state).detach())))
            else:
                next_qval = reward   
            
            pred_qval = pred_qval.to(torch.float32).unsqueeze(-1)
            #print('pred_qval', pred_qval, pred_qval.size())
            next_qval = next_qval.to(torch.float32)
            #print('next_qval', next_qval, next_qval.size())

            loss = F.mse_loss(pred_qval, next_qval)
            #print('loss', loss)
            writer.add_scalar('loss', loss, i)
            loss.backward()

            optimizer.step()
            print('step', step_idx, 'i', i, 'j', j)

            # Update the target network, copying all weights and biases in DQN
            # Periodically update the target network by Q network to target Q network
            if i % 200 == 0:
                #print('update weights', step_idx)
                # Update weights of target
                target_model.load_state_dict(policy_model.state_dict())

    writer.close()