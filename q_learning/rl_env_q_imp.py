import numpy as np
class Env:
    def __init__(self):

        self.width = 5
        self.height = 5
        self.reward = [[0.0] * self.width for _ in range(self.width)]
        self.done=False

    def reset(self,x=0,y=0):
        self.done = False
        return [x,y]

    def step(self, states, action):
        next_states = self.get_state(states, action)

        if next_states[0]==6 and next_states[1]==6:
            self.reward=5.0
            self.done=True
        elif (next_states[0]==2 and next_states[1]==1) or (next_states[0]==1 and next_states[1]==4) or (next_states[0]==2 and next_states[1]==5) or (next_states[0]==5 and next_states[1]==4)or (next_states[0]==6 and next_states[1]==1):
            self.reward = -3.0
            self.done = True
        else:
            self.reward=0
            self.done=False

        return next_states,self.reward,self.done

    def set_reward(self,obstacle,goal):
        self.reward[1][3]=obstacle
        self.reward[6][6]=goal

    def get_state(self,states, action):
        action_grid = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        new_states=[0,0]
        new_states[0] = states[0]+action_grid[action][0]
        new_states[1] = states[1]+action_grid[action][1]

        if new_states[0] < 0:
            new_states[0] = 0
        elif new_states[0] > 6:
            new_states[0] = 6

        if new_states[1] < 0:
            new_states[1] = 0
        elif new_states[1] > 6:
            new_states[1] = 6

        return new_states
