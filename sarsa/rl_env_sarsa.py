import numpy as np
class Env:
    def __init__(self):

        self.width = 5
        self.height = 5
        self.reward = [[0.0] * self.width for _ in range(self.width)]
        #self.state=[0,0]
        #self.next_states=[0,0]
        #self.reward[4][4] = 10.0  # (2,2) 좌표 동그라미 위치에 보상 1
        #self.reward[2][1] = -3.0  # (1,2) 좌표 세모 위치에 보상 -1
        #self.reward[1][3] = -3.0  # (2,1) 좌표 세모 위치에 보상 -1
        self.done=False

    def reset(self):
        self.done = False
        return [0,0]

    def step(self, states, action):
        #print('input', state)
        next_states = self.get_state(states, action)

        #print(action)
        #print('output',next_state)

        if next_states[0]==4 and next_states[1]==4:
            self.reward=5.0
            self.done=True
        elif next_states[0]==2 and next_states[1]==1:
            self.reward = -3.0
            self.done = True
        else:
            self.reward=0
            self.done=False

        return next_states,self.reward,self.done

    def set_reward(self,obstacle,goal):
        ##self.reward[2][1]=obstacle
        self.reward[1][3]=obstacle
        self.reward[4][4]=goal

    def get_state(self,states, action):
        action_grid = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        new_states=[0,0]
        new_states[0] = states[0]+action_grid[action][0]
        new_states[1] = states[1]+action_grid[action][1]

        if new_states[0] < 0:
            new_states[0] = 0
        elif new_states[0] > 4:
            new_states[0] = 4

        if new_states[1] < 0:
            new_states[1] = 0
        elif new_states[1] > 4:
            new_states[1] = 4

        return new_states
