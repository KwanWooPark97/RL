import numpy as np
from rl_env import Env

class Train:
    def __init__(self,env,grid_height,grid_width):

        self.env=env
        self.grid_width=grid_width
        self.grid_height=grid_height
        self.value_table=np.zeros([grid_height, grid_width], dtype=float)
        self.action= [0, 1, 2, 3]
        self.policy_table = [[[0.25, 0.25, 0.25, 0.25]] * grid_height
                             for _ in range(grid_width)]
        self.policy_table[4][4] = [0,0,0,0]
        self.discount=0.9

    def value_evaluation(self):
        next_value_table = np.zeros([self.grid_height, self.grid_width], dtype=float)
        action_match = ['Up', 'Down', 'Left', 'Right']
        action_table = []
        for i in range(self.grid_height):
            for j in range(self.grid_width):
                if i == j and (i == 4):
                    action_table.append('T')
                    next_value_table[4][4] = 0.0
                    continue
                else:
                    value_t = []
                    for act in self.action:
                        next_state,reward = env.step([i, j], act)
                        value = self.policy_table[i][j][act] * (reward + self.discount * self.value_table[next_state[0]][next_state[1]])
                        value_t.append(value)
                next_value_table[i][j] = round(max(value_t), 3)
                idx = np.argmax(value_t)
                action_table.append(action_match[idx])
        self.value_table = next_value_table
        print('Value Table:\n{}\n'.format(self.value_table))
        action_table = np.asarray(action_table).reshape((self.grid_height, self.grid_width))
        print('at each state, chosen action is :\n{}'.format(action_table))

    def run(self):
        self.env.set_reward(obstacle=-3.0,goal=10.0)
        for i in range(8):
            self.value_evaluation()


if __name__ == "__main__":
    env=Env()
    model=Train(env,5,5)
    model.run()
