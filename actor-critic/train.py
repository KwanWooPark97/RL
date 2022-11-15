import sys
import gym
import pylab
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.initializers import RandomUniform


# 정책 신경망과 가치 신경망 생성
class A2C(tf.keras.Model):#기본적인 내용을 다루는 a2c이기 때문에 크리틱과 액터를 구분하지 않음 보통은 따로 사용
    def __init__(self, action_size):
        super(A2C, self).__init__()
        self.actor_fc = Dense(24, activation='tanh')
        self.actor_out = Dense(action_size, activation='softmax',
                               kernel_initializer=RandomUniform(-1e-3, 1e-3))#액션의 출력을 확률로 반환
        self.critic_fc1 = Dense(24, activation='tanh')
        self.critic_fc2 = Dense(24, activation='tanh')
        self.critic_out = Dense(1,
                                kernel_initializer=RandomUniform(-1e-3, 1e-3))#크리틱의 출력은 V 함수를 의미함

    def call(self, x):
        actor_x = self.actor_fc(x)
        policy = self.actor_out(actor_x)

        critic_x = self.critic_fc1(x)
        critic_x = self.critic_fc2(critic_x)
        value = self.critic_out(critic_x)
        return policy, value


# 카트폴 예제에서의 액터-크리틱(A2C) 에이전트
class A2CAgent:
    def __init__(self, action_size):
        self.render = False#카트폴이 학습하는 모습을 보고싶으면 True로 바꿈

        # 행동의 크기 정의
        self.action_size = action_size

        # 액터-크리틱 하이퍼파라미터
        self.discount_factor = 0.99
        self.learning_rate = 0.001

        # 정책신경망과 가치신경망 생성
        self.model = A2C(self.action_size)
        # 최적화 알고리즘 설정, 미분값이 너무 커지는 현상을 막기 위해 clipnorm 설정, Adam 말고 RMSprop나 다른 옵티마이저도 사용해보는 경험 만들기
        self.optimizer = Adam(lr=self.learning_rate, clipnorm=5.0)

    # 정책신경망의 출력을 받아 확률적으로 행동을 선택
    def get_action(self, state):
        policy, _ = self.model(state)
        policy = np.array(policy[0])
        return np.random.choice(self.action_size, 1, p=policy)[0]#액터에서 나온 출력은 모든 액션에 대한 확률이므로 그 확률에 의해서 한개를 선택해주는 넘파이 함수

    # 각 타임스텝마다 정책신경망과 가치신경망을 업데이트
    def train_model(self, state, action, reward, next_state, done):
        model_params = self.model.trainable_variables#학습할 네트워크의 변수들을 가져옴
        with tf.GradientTape() as tape:
            policy, value = self.model(state)#현재상태의 정책과 가치함수를 가져옴
            _, next_value = self.model(next_state)#다음 상태의 가치함수를 가져옴
            target = reward + (1 - done) * self.discount_factor * next_value[0]#MSE에서 사용할 정답부분

            # 정책 신경망 오류 함수 구하기
            one_hot_action = tf.one_hot([action], self.action_size)#액션을 원핫벡터로 바꿈
            action_prob = tf.reduce_sum(one_hot_action * policy, axis=1)#위에 정책신경망에서 나온 정책을 곱해서 액션 확률로 만듬
            cross_entropy = - tf.math.log(action_prob + 1e-5)#크로스엔트로피 함수
            advantage = tf.stop_gradient(target - value[0])#정책신경망의 오류함수이기 때문에 가치신경망을 업데이트 하지 않기위해서
            actor_loss = tf.reduce_mean(cross_entropy * advantage)#정책신경망의 오류함수

            # 가치 신경망 오류 함수 구하기
            critic_loss = 0.5 * tf.square(tf.stop_gradient(target) - value[0])#MSE 오류함수
            critic_loss = tf.reduce_mean(critic_loss)

            # 하나의 오류 함수로 만들기
            loss = 0.2 * actor_loss + critic_loss#두 오류함수를 더하는대 정책신경망의 비중을 줄여줌

        # 오류함수를 줄이는 방향으로 모델 업데이트
        grads = tape.gradient(loss, model_params)#loss를 이용해서 학습하는 변수들의 미분값을 구함
        self.optimizer.apply_gradients(zip(grads, model_params))#미분값을 이용하고 옵티마이저를 통해서 역전파를 적용
        return np.array(loss)#텐서보드를 통해 loss를 확인하기 위한 반환값 보통 반환을 잘 안하는데 학습 과정을 보고싶으면 설정해둠


if __name__ == "__main__":
    # CartPole-v1 환경, 최대 타임스텝 수가 500
    env = gym.make('CartPole-v1')
    # 환경으로부터 상태와 행동의 크기를 받아옴
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n

    # 액터-크리틱(A2C) 에이전트 생성
    agent = A2CAgent(action_size)

    scores, episodes = [], []
    score_avg = 0

    num_episode = 1000
    for e in range(num_episode):
        done = False
        score = 0
        loss_list = []
        state = env.reset()
        state = np.reshape(state, [1, state_size])

        while not done:
            if agent.render:
                env.render()#위에서 설정한 학습하는 과정을 볼지말지 정해주는 코드

            action = agent.get_action(state)
            next_state, reward, done, info = env.step(action)#step을 하여 학습에 필요한 데이터 수집
            next_state = np.reshape(next_state, [1, state_size])#네트워크 입력으로 사용하기 위해 batch_size를 의미하는 1차원 추가

            # 타임스텝마다 보상 0.1, 에피소드가 중간에 끝나면 -1 보상
            score += reward
            reward = 0.1 if not done or score == 500 else -1

            # 매 타임스텝마다 학습
            loss = agent.train_model(state, action, reward, next_state, done)
            loss_list.append(loss)
            state = next_state

            if done:
                # 에피소드마다 학습 결과 출력
                score_avg = 0.9 * score_avg + 0.1 * score if score_avg != 0 else score
                print("episode: {:3d} | score avg: {:3.2f} | loss: {:.3f}".format(
                      e, score_avg, np.mean(loss_list)))

                # 에피소드마다 학습 결과 그래프로 저장
                scores.append(score_avg)
                episodes.append(e)
                pylab.plot(episodes, scores, 'b')
                pylab.xlabel("episode")
                pylab.ylabel("average score")
                pylab.savefig("./save_graph/graph.png")

                # 이동 평균이 400 이상일 때 종료
                if score_avg > 400:
                    agent.model.save_weights("./save_model/model", save_format="tf")#학습된 모델 저장, save_model을 사용하지 않음 weights를 사용함
                    sys.exit()#시스템 종료