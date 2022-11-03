import numpy as np
from rl_env_sarsa import Env
from collections import defaultdict
import random

class Train:
    def __init__(self,env,grid_height,grid_width):

        self.env=env
        self.grid_width=grid_width
        self.grid_height=grid_height
        self.action= [0, 1, 2, 3]
        #self.Q_table = np.zeros(((4,self.grid_width, self.grid_height)))
        self.discount=0.99
        self.epsilon=0.1
        self.learning_rate=0.01
        #self.state=[0.0,0.0]
        self.Q_table = defaultdict(lambda: [0.0, 0.0, 0.0, 0.0])
        self.action_match = ['Up', 'Down', 'Left', 'Right']

    def arg_max(self,q_list):
        max_idx_list = np.argwhere(q_list == np.amax(q_list))
        max_idx_list = max_idx_list.flatten().tolist()
        return random.choice(max_idx_list)

    def get_action(self, state):
        if np.random.rand() < self.epsilon:
            # 무작위 행동 반환
            action = np.random.choice(self.action)
        else:
            # 큐함수에 따른 행동 반환
            state = str(state)
            q_list = self.Q_table[state]
            action = self.arg_max(q_list)

        return action


    def update(self, state, action, reward, next_state, next_action,done):
        state, next_state = str(state), str(next_state)  # dictionary 에서 key는 str만 받으므로 형변환
        current_q = self.Q_table[state][action]  # 현재 상태에서 액션을 했을때 q함수값
        next_state_q = self.Q_table[next_state][next_action]  # 다음 상태에서 다음 액션을 했을때 q함수값
        td = reward + (self.discount * next_state_q*(1.0-done)) - current_q  # 시간차 에러(Temporal-Difference Error) terminal 상황에서는 done을 이용해 q_{t+1}을 제거해줌
        new_q = current_q + self.learning_rate * td  # 살사의 업데이트 식
        self.Q_table[state][action] = new_q

    def save_actionseq(self, action_sequence, action):
        #idx = self.action_grid.index(action)
        action_sequence.append(self.action_match[action])

    def run(self):
        total_episode = 100


        for episode in range(total_episode):
            action_sequence = []
            total_reward = 0

            # initial state, action, done
            state = self.env.reset()


            while True:

                action = self.get_action(state)
                self.save_actionseq(action_sequence, action)
                # next state, action
                next_state, reward, done = env.step(state, action)
                next_action = self.get_action(next_state)

                self.update(state, action, reward, next_state, next_action, done)
                total_reward += reward

                if done:
                    print('finished at', state, 'reward:', reward)
                    print('episode :{}, The sequence of action is:\
                                         {}'.format(episode, action_sequence))
                    print('Q-table:\n', dict(self.Q_table), '\n')
                    break
                else:
                    state = next_state

if __name__ == '__main__':
    env = Env()
    model = Train(env, 5, 5)
    model.run()




