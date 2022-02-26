from tensorboardX import SummaryWriter
from environments.env import SchedulerEnv

if __name__ == "__main__":
    #initialise environment, model and optimiser
    env = SchedulerEnv()
    writer = SummaryWriter()

    for i in range(1000):

        state = env.reset()
        done = False
        eps_reward = 0
        eps_steps = 0

        while not done:
            
            #generates random action from action space
            action = env.action_space.sample()

            #run through step to book appointment
            new_state, reward, done, info = env.step(action)

            eps_reward += reward
            eps_steps += 1
            
            writer.add_scalar("eps_reward", eps_reward, i )

            #update state to new state
            state = new_state
        
        if i % 100 == 0:
            print('episode_reward', eps_reward, 'step', i)

    writer.close()