import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras import optimizers, losses
from tensorflow.keras import Model
import numpy as np
import random
from collections import deque
import gym

env = gym.make('CartPole-v0')#카트폴 환경을 가져옴
num_action = env.action_space.n#액션의 사이즈를 정의해줌
state_size = env.observation_space.shape[0]#state의 사이즈를 정의해줌


class Duel_DQN(Model):  # Q 함수를 추정하기 위해 만든 dqn 뉴런 네트워크입니다. 입력은 state며 출력은 Q함수입니다.
    def __init__(self):
        super(Duel_DQN, self).__init__()
        self.layer1 = Dense(64, activation='relu')
        self.layer2 = Dense(64, activation='relu')
        self.value = Dense(1) #V 함수 값 추정을 위해 만든 hidden layer 입니다.
        self.advantage=Dense(num_action) #Advantage 함수 값 추정을 위해 만든 hidden layer 입니다.

    def call(self, state):
        layer1 = self.layer1(state)
        layer2 = self.layer2(layer1)
        value = self.value(layer2)#하나의 layer 출력을 이용하여 2개의 입력으로 사용
        advantage=self.advantage(layer2)
        result=tf.add(value, advantage) # 위에서 구한 V 함수값과 A 함수 값을 더해서 Q 함수를 만들어 줍니다.
        return result


class DQNtrain:
    def __init__(self):
        # hyper parameters
        self.lr = 0.001  # learning rate
        self.df = 0.99  # discount factor

        self.dqn_model = Duel_DQN()  # original network
        self.dqn_target = Duel_DQN()  # target network
        self.opt = optimizers.RMSprop(lr=self.lr, )  # optimizer를 RMSprop를 사용했습니다 Adam으로 해도 좋을 것 같습니다.

        self.epsilon = 1.0  # 탐험을 위한 앱실론
        self.epsilon_decay = 0.999  # 앱실론이 줄어드는 비율
        self.epsilon_min = 0.01  # 앱실론의 최소값
        self.batch_size = 64  # 학습을 위한 batch size
        self.train_start = 1000  # 학습을 위해 데이터 준비를 몇개 할지 정해줍니다.
        self.state_size = state_size

        self.memory = deque(maxlen=2000)  # 버퍼를 만드는데 2000개가 넘어가면 오래된것부터 지워버립니다.

        # tensorboard 이 밑은 tensorboard를 사용하기 위한 내용들입니다.
        self.log_dir = 'logs/'
        self.train_summary_writer = tf.summary.create_file_writer(self.log_dir)
        self.reward_board = tf.keras.metrics.Mean('reward_board', dtype=tf.float32)

    def update_target(self):
        self.dqn_target.set_weights(
            self.dqn_model.get_weights())  # original network의 가중치를 가져와서 target network에 붙여넣기 해줍니다.

    def get_action(self, state):
        if np.random.rand() <= self.epsilon:  # 앱실론 비율에 따라서 action을 랜덤으로 해줍니다.
            return random.randrange(num_action)
        else:
            q_value = self.dqn_model(tf.convert_to_tensor(state[None, :],
                                                          dtype=tf.float32))  # original network의 출력으로 나온 Q 함수 중에서 가장 큰 값을 가지는 action을 취해줍니다.
            return np.argmax(q_value[0])

    def append_sample(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))  # 버퍼에 학습을 위한 샘플들을 저장합니다.

    def train(self):
        if self.epsilon > self.epsilon_min:  # 탐험의 비율을 학습을 진행하며 점점 줄여줍니다.
            self.epsilon *= self.epsilon_decay

        mini_batch = random.sample(self.memory, self.batch_size)  # 버퍼에서 학습을 위한 미니 배치 데이터를 가져옵니다.

        states = np.zeros((self.batch_size, self.state_size))
        next_states = np.zeros((self.batch_size, self.state_size))
        actions, rewards, dones = [], [], []

        for i in range(self.batch_size):
            states[i] = mini_batch[i][0]
            actions.append(mini_batch[i][1])
            rewards.append(mini_batch[i][2])
            next_states[i] = mini_batch[i][3]
            dones.append(mini_batch[i][4])#학습에 사용할 각 변수에 데이터들을 저장합니다. unzip과 같은 역할을 수행함

        dqn_variable = self.dqn_model.trainable_variables#학습할 네트워크의 가중치들을 가져옵니다

        with tf.GradientTape() as tape:#학습 시작
            tape.watch(dqn_variable)

            target = self.dqn_model(tf.convert_to_tensor(np.vstack(states), dtype=tf.float32))  # original Q 함수 값을 구합니다.
            # target_action = self.dqn_model(tf.convert_to_tensor(np.vstack(states), dtype=tf.float32))
            target_val = self.dqn_target(
                tf.convert_to_tensor(np.vstack(next_states), dtype=tf.float32))  # target Q 함수 값을 구합니다.

            target = np.array(target)
            # target_action = np.array(target)
            target_val = np.array(target_val)#계산을 빠르게 하기위해 numpy 형태로 바꿔줍니다.

            for i in range(self.batch_size):
                best_action = np.argmax(target[i])  # Q 함수값중 가장 큰 값을 가지는 action을 구합니다.
                next_v = target_val[
                    i, best_action]  # 학습하는 과정은 DDQN과 동일합니다.
                if dones[i]:
                    target[i][actions[i]] = rewards[i]
                else:
                    target[i][actions[i]] = rewards[i] + self.df * next_v  # TD 값을 구해줍니다.

            values = self.dqn_model(tf.convert_to_tensor(np.vstack(states), dtype=tf.float32))
            error = tf.square(values - target) * 0.5  # MSE error를 이용합니다.
            error = tf.reduce_mean(error)

        dqn_grads = tape.gradient(error, dqn_variable)  # 에러 값의 gradient를 구해줍니다.
        self.opt.apply_gradients(zip(dqn_grads, dqn_variable))  # 구한 gradient를 이용해 학습을 진행합니다.

    def run(self):

        t_end = 500
        epi = 100000

        state = env.reset()#환경 초기화
        state = np.reshape(state, [1, state_size])#네트워크 입력에 넣기위해 차원을 추가해줌

        for e in range(epi):
            total_reward = 0
            for t in range(t_end):
                action = self.get_action(state)
                next_state, reward, done, info = env.step(action)#환경과 상호작용하여 1 step 움직임
                next_state = np.reshape(next_state, [1, state_size])#네트워크 입력에 넣기위해 차원을 추가함

                if t == t_end:#terminal 상태 확인
                    done = True
                if t < t_end and done:
                    reward = -1

                total_reward += reward
                self.append_sample(state, action, reward, next_state, done)#버퍼에 데이터 추가

                if len(self.memory) >= self.train_start:#버퍼에 일정량 데이터가 쌓일때까지 학습안함
                    self.train()

                total_reward += reward
                state = next_state

                if done:#에피소드가 끝나면 타겟 네트워크 업데이트 및 텐서보드 확인을 위한 파일 저장
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
