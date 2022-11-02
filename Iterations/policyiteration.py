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

    def policy_evaluation(self):
        next_value_table = np.zeros([self.grid_height, self.grid_width], dtype=float)

        for i in range(self.grid_height):
            for j in range(self.grid_width):
                if i == j and (i == 4):
                    next_value_table[4][4] = 0.0
                    continue
                else:
                    value_t = 0.0
                    for act in self.action:
                        next_state,reward = env.step([i, j], act)
                        value = self.policy_table[i][j][act] * (reward + self.discount * self.value_table[next_state[0]][next_state[1]])
                        value_t += value
                next_value_table[i][j] = round(value_t, 3)

        self.value_table = next_value_table
        print('Value Table:\n{}\n'.format(self.value_table))


    def policy_improvement(self):

        next_policy = self.policy_table
        action_match = ['Up', 'Down', 'Left', 'Right']
        action_table = []

        # get Q-func.
        for i in range(self.grid_height):
            for j in range(self.grid_width):
                result = [0.0, 0.0, 0.0, 0.0]
                if i == j and (i == 4):
                    action_table.append('T')
                    continue
                else:
                    q_func_list = []
                    for k in range(len(self.action)):

                        next_state, reward = self.env.step([i, j], k)
                        value = reward + self.discount * self.value_table[next_state[0]][next_state[1]]
                        q_func_list.append(value)

                    max_idx_list = np.argwhere(q_func_list == np.amax(q_func_list))# value_list 에서 가장 큰값은 반환 여러개일경우를 대비해 argwhere로 여러개 다 반환 가능하게 함
                    max_idx_list = max_idx_list.flatten().tolist()
                    prob = 1 / len(max_idx_list)  # 여러개일경우 방향의 확률이 같게 만들어줌

                    for idx in max_idx_list:
                        result[idx] = prob

                    next_policy[i][j] = result
                    idx = np.argmax(next_policy[i][j])
                    action_table.append(action_match[idx])

        self.policy_table = next_policy
        action_table = np.asarray(action_table).reshape((self.grid_height, self.grid_width))

        print('Updated policy is :\n{}\n'.format(np.array(self.policy_table)))
        print('at each state, chosen action is :\n{}\n'.format(action_table))

    def run(self):
        self.env.set_reward(obstacle=-3.0,goal=10.0)
        for i in range(10):

            self.policy_evaluation()
            self.policy_improvement()

if __name__ == "__main__":
    env=Env()
    model=Train(env,5,5)
    model.run()
