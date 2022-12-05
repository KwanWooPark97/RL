import numpy as np
from rl_env_q_imp import Env
from collections import defaultdict
import random

class Train:
    def __init__(self,env,grid_height,grid_width):

        self.env=env
        self.grid_width=grid_width
        self.grid_height=grid_height
        self.action= [0, 1, 2, 3]
        self.discount=0.99
        self.epsilon=0.1
        self.learning_rate=0.01
        self.Q_table = defaultdict(lambda: [0.0, 0.0, 0.0, 0.0])# Q 테이블을 만들기위해 딕셔너리로 변환
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


    def update(self, state, action, reward, next_state,done):
        state, next_state = str(state), str(next_state)  # dictionary 에서 key는 str만 받으므로 형변환
        current_q = self.Q_table[state][action]  # 현재 상태에서 액션을 했을때 q함수값
        next_state_q = max(self.Q_table[next_state])  # 다음 상태에서 다음 액션을 했을때 q함수값 SARSA와 여기서 달라짐 action을 미리 알 필요가 없음
        td = reward + (self.discount * next_state_q*(1.0-done)) - current_q  # 시간차 에러(Temporal-Difference Error)
        new_q = current_q + self.learning_rate * td  # 살사의 업데이트 식
        self.Q_table[state][action] = new_q

    def save_actionseq(self, action_sequence, action):
        action_sequence.append(self.action_match[action])

    def run(self):
        total_episode = 200

        for episode in range(total_episode):
            action_sequence = []
            total_reward = 0

            # initial state, action, done
            state = self.env.reset()
            done = False

            while True:

                action = self.get_action(state)
                self.save_actionseq(action_sequence, action)

                next_state, reward, done = env.step(state, action)


                self.update(state, action, reward, next_state, done)
                total_reward += reward

                if done:
                    print('finished at', state ,'reward:',reward)
                    print('episode :{}, The sequence of action is:\
                     {}'.format(episode, action_sequence))
                    print('Q-table:\n',dict(self.Q_table),'\n')
                    break
                else:
                    state = next_state

    def test(self):
        print('-----------------------------test----------------------------------')
        total_episode = 5

        for episode in range(total_episode):
            action_sequence = []
            total_reward = 0

            # initial state, action, done
            random_start_x=np.random.randint(0, 6)
            random_start_y=np.random.randint(0, 6)
            state = self.env.reset(random_start_x,random_start_y)
            done = False

            while True:

                action = self.get_action(state)
                self.save_actionseq(action_sequence, action)

                next_state, reward, done = env.step(state, action)

                total_reward += reward

                if done:
                    print('finished at', state ,'reward:',reward)
                    print('episode :{}, The sequence of action is:\
                     {}'.format(episode, action_sequence))
                    print('start position:\n',random_start_y,random_start_x,'\n')
                    break
                else:
                    state = next_state

if __name__ == '__main__':
    env = Env()
    model = Train(env, 7, 7)
    model.run()
    model.test()



