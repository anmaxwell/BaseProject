import gym
import numpy as np
import pandas as pd


class SchedulerEnv(gym.Env):

    def __init__(self):
        
        #starting parameters
        num_gps = 100
        num_slots = 32
        num_pre_booked = 750
        #to_book = [2,1,2,2,1,1,1,3,3,1,2,1,3,2,1,1,2,1,3,2,3,2]
        to_book = [2,1,1,1,1]
        num_to_book = len(to_book)
        agent_pos = [0,0]
        
        #set parameters for the day
        self.num_gps = num_gps
        self.num_slots = num_slots
        self.num_pre_booked = num_pre_booked
        self.to_book = to_book
        self.num_to_book = num_to_book
        self.diary_slots = num_gps*num_slots
        self.agent_pos = agent_pos

        #set action space to move around the grid
        self.action_space = gym.spaces.Discrete(4) #up, down, left, right
        
        #set observation space 
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=(self.num_slots, self.num_gps), dtype=np.int32)
   
    #creates daily diary for each gp, randomly populates prebooked appointments and resets parameters
    def reset(self):

        #creates zero filled dataframe with row per time slot and column per gp
        self.state = np.zeros((self.num_slots, self.num_gps),dtype=float)

        #randomly enters a 1 for each pre booked appointments
        pre_booked = self.num_pre_booked
        while pre_booked>0:
            pre_booked -= 1
            self.state[np.random.randint(self.num_slots), np.random.randint(self.num_gps)] = 1
            
        #randomly sets the agent start space
        self.agent_pos = [np.random.randint(self.num_slots), np.random.randint(self.num_gps)]

        #resets parameters for new episode
        self.done = False
        self.reward = 0
        self.appt_idx = 0
        
        #print('starting state', self.state.sum(), self.state)

        return self.state
    
    #calculates new position of the agent based on the action
    def move_agent(self, action):

        #set boundaries for the grid
        max_row = self.num_slots - 1
        max_col = self.num_gps - 1

        #setting new co-ordinates for the agent
        new_row = self.agent_pos[0]
        new_col = self.agent_pos[1]

        #calculate what the new position may be based on the action without going out the grid
        if action == 0:
            #print('up')
            new_row = max(self.agent_pos[0] - 1, 0)
        if action == 1:
            #print('down')
            new_row = min(self.agent_pos[0] + 1, max_row)
        if action == 2:
            #print('left')
            new_col = max(self.agent_pos[1] - 1, 0)
        if action == 3:
            #print('right')
            new_col = min(self.agent_pos[1] + 1, max_col)

        new_pos = [new_row, new_col]
        #print('new pos', new_pos)

        return new_pos

    #checks if we can look to book appointment starting here
    def check_bookable(self):
        return self.state[self.agent_pos[0], self.agent_pos[1]] == 0.0
    
    #action if we can't book the appointment
    def invalid_booking(self):
        #print('cant book')
        self.reward = -1
        
    #action if we can book the appointment
    def valid_booking(self):
        #print('go ahead and book')
        self.appt_idx += 1
        self.reward = 1
    
    #checks if the appointment fits
    def check_and_book(self):
        
        max_row = self.num_slots - 1
        cells_to_check = self.to_book[self.appt_idx]
        
        if cells_to_check==1:
            #print('good to check for single')
            if self.state[self.agent_pos[0], self.agent_pos[1]] == 0:
                self.state[self.agent_pos[0], self.agent_pos[1]] = 1
                self.valid_booking()
            else:
                #print('single taken')
                self.invalid_booking()

        if cells_to_check==2:
            #check we're not at the bottom of the grid
            if self.agent_pos[0]<max_row:
                #check the next cells is also 0.0
                #print('good to check for double')
                if self.state[self.agent_pos[0], self.agent_pos[1]] == 0 and \
                self.state[(self.agent_pos[0]+1), self.agent_pos[1]] == 0:
                    self.state[self.agent_pos[0], self.agent_pos[1]] = 1
                    self.state[(self.agent_pos[0]+1), self.agent_pos[1]] = 1
                    self.valid_booking()
                    self.agent_pos = [(self.agent_pos[0]+1), self.agent_pos[1]]
                    #print('after booking', self.agent_pos)
                else:
                    #print('double taken')
                    self.invalid_booking()
            else:
                #print('not for double')
                self.invalid_booking()
                
        if cells_to_check==3:
            #check we're not at the bottom of the grid
            if self.agent_pos[0]+1<max_row:
                #print('good to check for treble')
                if self.state[self.agent_pos[0], self.agent_pos[1]] == 0 and \
                self.state[(self.agent_pos[0]+1), self.agent_pos[1]] == 0 \
                 and self.state[(self.agent_pos[0]+2), self.agent_pos[1]] == 0:
                    self.state[self.agent_pos[0], self.agent_pos[1]] = 1
                    self.state[(self.agent_pos[0]+1), self.agent_pos[1]] = 1
                    self.state[(self.agent_pos[0]+2), self.agent_pos[1]] = 1
                    self.valid_booking()
                    self.agent_pos = [(self.agent_pos[0]+2), self.agent_pos[1]]
                else:
                    #print('treble taken')
                    self.invalid_booking()
            else:
                #print('not for treble')
                self.invalid_booking()
                
        if cells_to_check==4:
            #check we're not at the bottom of the grid
            if self.agent_pos[0]+2<max_row:
                #check the next cells is also 0.0
                #print('good for quad')
                if self.state[self.agent_pos[0], self.agent_pos[1]] == 0 and \
                self.state[(self.agent_pos[0]+1), self.agent_pos[1]] == 0 \
                 and self.state[(self.agent_pos[0]+2), self.agent_pos[1]] == 0 and \
                self.state[(self.agent_pos[0]+3), self.agent_pos[1]] == 0:
                    self.state[self.agent_pos[0], self.agent_pos[1]] = 1
                    self.state[(self.agent_pos[0]+1), self.agent_pos[1]] = 1
                    self.state[(self.agent_pos[0]+2), self.agent_pos[1]] = 1
                    self.state[(self.agent_pos[0]+3), self.agent_pos[1]] = 1
                    self.valid_booking()
                    self.agent_pos = [(self.agent_pos[0]+3), self.agent_pos[1]]
                else:
                    #print('quad taken')
                    self.invalid_booking()
            else:
                #print('not for quad')
                self.invalid_booking()

        next_state = self.state

        return next_state

    def step(self, action):

        #get new position of agent based on action
        new_agent_pos = self.move_agent(action)
        #print('new and old pos', new_agent_pos, self.agent_pos)
        
        #if the agent is stuck on an edge then move to a new position
        if new_agent_pos == self.agent_pos:
            self.agent_pos = [np.random.randint(self.num_slots), np.random.randint(self.num_gps)]
            #print('here1', self.agent_pos)
        else:
            self.agent_pos = new_agent_pos
            #print('here2', self.agent_pos)
            
        #print('trying to book', self.to_book, self.appt_idx)
        
        #check if it's possible to book then book
        if self.check_bookable():
            #print('checked here')
            self.state = self.check_and_book()
        else:
            #print('not bookable')
            self.invalid_booking()
        
        #work out if episode complete
        if self.appt_idx == len(self.to_book):
            self.done = True
            
        #print(self.state, self.agent_pos)
        agent_state = self.state.copy()
        agent_state[self.agent_pos[0], self.agent_pos[1]] = 5
        #print('agent', agent_state)

        info = {}
        return agent_state, self.reward, self.done, info