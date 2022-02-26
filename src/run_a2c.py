import numpy as np
import ptan
#import argparse
from tensorboardX import SummaryWriter
import torch
import torch.optim as optim
import torch.nn.functional as F

from environments.env import SchedulerEnv
from models.a2c import Model


gamma = 0.99
batch_size = 11
num_envs = 6
reward_steps = 4

def unpack_batch(batch, model, device='cpu'):

    states = []
    actions = []
    rewards = []
    not_done_idx = []
    last_states = []
    #create lists of the states, actions and rewards
    for idx, exp in enumerate(batch):
        states.append(np.array(exp.state, copy=False))
        actions.append(int(exp.action))
        rewards.append(exp.reward)
        #separate out the last states to be able to calculate the rewards
        if exp.last_state is not None:
            not_done_idx.append(idx)
            last_states.append(np.array(exp.last_state, copy=False))

    #convert to tensors for calculations
    states = torch.FloatTensor(
        np.array(states, copy=False)).to(device)
    actions = torch.LongTensor(actions).to(device)

    # handle rewards
    rewards_np = np.array(rewards, dtype=np.float32)
    if not_done_idx:
        last_states = torch.FloatTensor(np.array(last_states, copy=False)).to(device)
        last_vals = model(last_states)[1]
        last_vals_np = last_vals.data.cpu().numpy()[:, 0]
        last_vals_np *= gamma ** reward_steps
        rewards_np[not_done_idx] += last_vals_np

    rewards = torch.FloatTensor(rewards_np).to(device)

    return states, actions, rewards


if __name__ == "__main__":
    #device = "cuda"
    device = "cpu"

    #create multiple environments for multiprocessing
    make_env = lambda: SchedulerEnv()
    envs = [make_env() for _ in range(num_envs)]

    #start writing to tensorboard
    writer = SummaryWriter(comment="Scheduler A2C")

    #initialise model, agent and run through episodes to get experience
    model = Model(envs[0].observation_space.shape, envs[0].action_space.n).to(device)
    agent = ptan.agent.PolicyAgent(lambda x: model(x)[0], apply_softmax=True, device=device)
    exp_source = ptan.experience.ExperienceSourceFirstLast(envs, agent, gamma=gamma, steps_count=reward_steps)

    optimizer = optim.Adam(model.parameters(), lr=0.001, eps=1e-3)

    #create list to capture batches
    batch = []

    #create lists to be used to record values for tracking averages
    reward_stack = []
    loss_stack = []

    #work through each experience source to capture state, actions etc
    for step_idx, exp in enumerate(exp_source):
        batch.append(exp)

        if len(batch) < batch_size:
            continue

        states, actions, rewards = unpack_batch(batch, model, device=device)
        batch.clear()

        optimizer.zero_grad()

        # using the network to give actions and state_value
        actor_val, critic_val = model(states)
        # [CRITIC] calculate the loss between value_state (just predicted now) and reward from the batch
        critic_loss = F.mse_loss(critic_val.squeeze(-1), rewards)

        # Runs the log_softmax against actor output (just predicted now)
        log_prob = F.log_softmax(actor_val, dim=1)
        # Advantage equals reward from the batch (size:[batch_size]) minus the value_state (just predicted now)
        advantage = rewards - critic_val.detach()

        # multiples the advantage at each step by the log probability of the chosen action for that step
        log_prob_actions = advantage * log_prob[range(batch_size), actions]
        # calculate the policy gradient adjustment to make (negated to move toward policy improvement)
        actor_loss = -log_prob_actions.mean()

        # perform softmax on action estimates (from ACTOR) (just predicted now)
        prob_val = F.softmax(actor_val, dim=1)
        # calculating the action entropy 
        entropy_loss = 0.01 * (prob_val * log_prob).sum(dim=1).mean()

        # calculate policy gradients only

        # [ACTOR] backpropogate
        actor_loss.backward(retain_graph=True)

        # apply entropy and value gradients
        # [CRITIC] backpropagate and apply entropy
        loss = entropy_loss + critic_loss
        loss.backward()

        optimizer.step()

        #send average loss and rewards to tensorboard
        if len(reward_stack) > 0 and step_idx % 10 == 0:
            #print(step_idx)
            avg_rewards = np.mean(reward_stack)
            avg_loss = np.mean(loss_stack)
            writer.add_scalar('ave_batch_reward', avg_rewards, step_idx)
            writer.add_scalar('ave_batch_loss', avg_loss, step_idx)
            print('ave_batch_reward', avg_rewards, 'step', step_idx)
            print('ave_batch_loss', avg_loss, 'step', step_idx)
            reward_stack.clear()
            loss_stack.clear()
        else:
            reward_stack.append(torch.mean(rewards).item())
            loss_stack.append(torch.mean(critic_loss).item())
            
        if step_idx > 500000:
            break

    writer.close()