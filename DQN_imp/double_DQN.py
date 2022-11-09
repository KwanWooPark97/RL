import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras import optimizers, losses
from tensorflow.keras import Model
import numpy as np
import random
from collections import deque
import gym

env = gym.make('CartPole-v0')
num_action = env.action_space.n
state_size = env.observation_space.shape[0]

class DDQN(Model):#Q 함수를 추정하기 위해 만든 dqn 뉴런 네트워크입니다. 입력은 state며 출력은 Q함수입니다.
    def __init__(self):
        super(DDQN, self).__init__()
        self.layer1 = Dense(64, activation='relu')
        self.layer2 = Dense(64, activation='relu')
        self.value = Dense(num_action)

    def call(self, state):
        layer1 = self.layer1(state)
        layer2 = self.layer2(layer1)
        value = self.value(layer2)
        return value


class DQNtrain:
    def __init__(self):
        # hyper parameters
        self.lr =0.001 #learning rate
        self.df = 0.99 #discount factor

        self.dqn_model = DDQN() #original network
        self.dqn_target = DDQN() #target network
        self.opt = optimizers.RMSprop(lr=self.lr, )#optimizer를 RMSprop를 사용했습니다 Adam으로 해도 좋을 것 같습니다.

        self.epsilon = 1.0 #탐험을 위한 앱실론
        self.epsilon_decay = 0.999 #앱실론이 줄어드는 비율
        self.epsilon_min = 0.01 #앱실론의 최소값
        self.batch_size = 64 #학습을 위한 batch size
        self.train_start = 1000 #학습을 위해 데이터 준비를 몇개 할지 정해줍니다.
        self.state_size = state_size

        self.memory = deque(maxlen=2000) #버퍼를 만드는데 2000개가 넘어가면 오래된것부터 지워버립니다.

        # tensorboard 이 밑은 tensorboard를 사용하기 위한 내용들입니다.
        self.log_dir = 'logs/'
        self.train_summary_writer = tf.summary.create_file_writer(self.log_dir)
        self.reward_board = tf.keras.metrics.Mean('reward_board', dtype=tf.float32)

    def update_target(self):
        self.dqn_target.set_weights(self.dqn_model.get_weights()) #original network의 가중치를 가져와서 target network에 붙여넣기 해줍니다.

    def get_action(self, state):
        if np.random.rand() <= self.epsilon: #앱실론 비율에 따라서 action을 랜덤으로 해줍니다.
            return random.randrange(num_action)
        else:
            q_value = self.dqn_model(tf.convert_to_tensor(state[None, :], dtype=tf.float32)) #original network의 출력으로 나온 Q 함수 중에서 가장 큰 값을 가지는 action을 취해줍니다.
            return np.argmax(q_value[0])

    def append_sample(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done)) #버퍼에 학습을 위한 샘플들을 저장합니다.

    def train(self):
        if self.epsilon > self.epsilon_min: #탐험의 비율을 학습을 진행하며 점점 줄여줍니다.
            self.epsilon *= self.epsilon_decay

        mini_batch = random.sample(self.memory, self.batch_size) #버퍼에서 학습을 위한 미니 배치 데이터를 가져옵니다.

        states = np.zeros((self.batch_size, self.state_size))
        next_states = np.zeros((self.batch_size, self.state_size))
        actions, rewards, dones = [], [], []

        for i in range(self.batch_size):
            states[i] = mini_batch[i][0]
            actions.append(mini_batch[i][1])
            rewards.append(mini_batch[i][2])
            next_states[i] = mini_batch[i][3]
            dones.append(mini_batch[i][4])

        dqn_variable = self.dqn_model.trainable_variables

        with tf.GradientTape() as tape:
            tape.watch(dqn_variable)

            target = self.dqn_model(tf.convert_to_tensor(np.vstack(states), dtype=tf.float32)) #original Q 함수 값을 구합니다.
            #target_action = self.dqn_model(tf.convert_to_tensor(np.vstack(states), dtype=tf.float32))
            target_val = self.dqn_target(tf.convert_to_tensor(np.vstack(next_states), dtype=tf.float32)) #target Q 함수 값을 구합니다.
            
            target = np.array(target)
            #target_action = np.array(target)
            target_val = np.array(target_val)
            
            for i in range(self.batch_size):
                best_action = np.argmax(target[i]) #Q 함수값중 가장 큰 값을 가지는 action을 구합니다.
                next_v = target_val[i, best_action] # 여기서 DQN과 다른 점이 나옵니다. 사용하는 action을 max action을 사용하는 것이 아닌 original Q 함수에서 max값을 가지는 action을 취합니다.
                if dones[i]:
                    target[i][actions[i]] = rewards[i]
                else:
                    target[i][actions[i]] = rewards[i] + self.df * next_v #TD 값을 구해줍니다.

            values = self.dqn_model(tf.convert_to_tensor(np.vstack(states), dtype=tf.float32))
            error = tf.square(values - target) * 0.5 #MSE error를 이용합니다.
            error = tf.reduce_mean(error)

        dqn_grads = tape.gradient(error, dqn_variable) #에러 값의 gradient를 구해줍니다.
        self.opt.apply_gradients(zip(dqn_grads, dqn_variable)) # 구한 gradient를 이용해 학습을 진행합니다.

    def run(self):

        t_end = 500
        epi = 100000

        state = env.reset()
        state = np.reshape(state, [1, state_size])

        for e in range(epi):
            total_reward = 0
            for t in range(t_end):
                action = self.get_action(state)
                next_state, reward, done, info = env.step(action)
                next_state = np.reshape(next_state, [1, state_size])

                if t == t_end :
                    done = True
                if t < t_end and done :
                    reward = -1

                total_reward += reward
                self.append_sample(state, action, reward, next_state, done)

                if len(self.memory) >= self.train_start:
                    self.train()

                total_reward += reward
                state = next_state

                if done:
                    self.update_target()
                    self.reward_board(total_reward)
                    print("e : ", e, " reward : ", total_reward, " step : ", t)
                    env.reset()
                    with self.train_summary_writer.as_default():
                        tf.summary.scalar('reward', total_reward, step=e)
                    break


if __name__ == '__main__':
    DQN = DQNtrain()
    DQN.run()
