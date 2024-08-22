# PPO(Proximal Policy Optimization)
PPO는 강화 학습에서 널리 사용되는 정책 기반(policy gradient) 알고리즘으로, 안정성(stability)과 효율성(efficiency)을 동시에 추구합니다. PPO는 기본적으로 기존의 정책을 크게 변경하지 않으면서, 현재의 정책을 점진적으로 업데이트하는 접근 방식을 채택합니다. 이 접근은 Trust Region Policy Optimization(TRPO) 알고리즘에서 유래했습니다.  

위에는 GPT에게 물어봤을 때 PPO의 정의이다.  
나는 실제로 PPO를 사용하여 테트리스 논문을 작성하면서 PPO를 정말 많이 공부했었다.  
여기서는 개념을 정리할 예정이므로 위 프로젝트를 진행하는 과정에서 힘들었던건 나중에 다루겠다.  

먼저 PPO의 motivation이다.  

![https://github.com/KwanWooPark97/RL_basic/blob/master/ch6_tf2_ppo/img/ppo_1.png](https://github.com/KwanWooPark97/RL_basic/blob/master/ch6_tf2_ppo/img/ppo_1.png)
on-policy 방식이므로 $\theta$가 업데이트 되면 기존에 있던 궤적들은 전부 폐기한다. (on-policy의 샘플 비효율성)  

Importance sampling  
* Importance Sampling은 확률분포를 추정할 때 사용하는 기법으로, 특히 강화 학습과 같은 확률론적 환경에서 자주 사용됩니다.
  이는 특정 분포로부터 직접 샘플링하는 것이 어려운 상황에서, 다른 분포로부터 샘플링한 결과를 활용해 기대값이나 확률을 계산하는 방법입니다.
내가 이해한 바로는 위에서 말한 샘플 비효율성을 줄이기위해 데이터를 폐기하지 말고 IS를 이용하여 폐기전에 업데이트를 진행하고 버린다.  

![https://github.com/KwanWooPark97/RL_basic/blob/master/ch6_tf2_ppo/img/ppo_2.png](https://github.com/KwanWooPark97/RL_basic/blob/master/ch6_tf2_ppo/img/ppo_2.png)  
![https://github.com/KwanWooPark97/RL_basic/blob/master/ch6_tf2_ppo/img/ppo_3.png](https://github.com/KwanWooPark97/RL_basic/blob/master/ch6_tf2_ppo/img/ppo_3.png)    
식 유도 과정 자체는 어렵지 않게 위에서 아래로 천천히 따라가면 의미를 이해할 수 있다.  
중간에 $\prod_{k=0}^{t}$ 이 부분이 갑자기 사라지는데 이유를 계속 고민하다 보니 마르코프 연쇄 때문에 이렇게 된거라고 생각한다.  
대충 t+1의 뭔가를 계산할때는 현재 t의 상황만 생각한다 그 성질이다.  
대신 밑에 사진을 보면 $\theta$가 변해도 **상태 전이 확률은 크게 변하면 안된다.** 최대한 비슷하다는 의미로 1이라는 가정을 하고 있기 때문에 이 가정이 무너지면 전체적인 loss가 변하게된다.  


![https://github.com/KwanWooPark97/RL_basic/blob/master/ch6_tf2_ppo/img/ppo_4.png](https://github.com/KwanWooPark97/RL_basic/blob/master/ch6_tf2_ppo/img/ppo_4.png)   

당연하지만 업데이트를 진행하면 항상 값은 커져야하므로 부등호가 성립한다.  
이 부등호에 맞게 식을 직접 넣고 식을 유도한다.  
![https://github.com/KwanWooPark97/RL_basic/blob/master/ch6_tf2_ppo/img/ppo_5.png](https://github.com/KwanWooPark97/RL_basic/blob/master/ch6_tf2_ppo/img/ppo_5.png)  
여기서 가장 중요한 점은 파란색 식을 유도한 마지막 결과를 보면 $\theta_{OLD}$의 V 함수값의 기대값이지만 궤적 자체는 $\theta$를 통해 얻었다는 것이다.
![https://github.com/KwanWooPark97/RL_basic/blob/master/ch6_tf2_ppo/img/ppo_6.png](https://github.com/KwanWooPark97/RL_basic/blob/master/ch6_tf2_ppo/img/ppo_6.png)  
무한 경계 상황을 가정하면서 V 함수 정의를 이용해 유도를 계속해간다.  
마지막을 보면 결국 DQN에서 봤었던 Advantage 함수를 이용해서 정의한다.  
A 함수를 생각해보면 이 식은 결국 $J-J_{OLD}$의 부등호에서 시작된거다. 그러면 A의 부호가 중요해진다.   
A가 0보다 크면 이는 평균적으로 이 행동이 괜찮다는 의미로 이 행동을 자주 하도록 업데이트한다.  
A가 0보다 작으면 이는 평균적으로 이 행동이 나쁘다는 의미로 이 행동을 지양 하도록 업데이트한다.  
![https://github.com/KwanWooPark97/RL_basic/blob/master/ch6_tf2_ppo/img/ppo_7.png](https://github.com/KwanWooPark97/RL_basic/blob/master/ch6_tf2_ppo/img/ppo_7.png)  
여기서는 베이즈 정리? (이건 확실하지 않음)를 사용하여 확률을 분해시켜서 계산을 편하게 만들어버린다.  
![https://github.com/KwanWooPark97/RL_basic/blob/master/ch6_tf2_ppo/img/ppo_8.png](https://github.com/KwanWooPark97/RL_basic/blob/master/ch6_tf2_ppo/img/ppo_8.png)  
위에서 유도한 모든 내용을 정리하면 결국 우리가 원하는 강화학습 기대값 loss를 계산 할 수 있다.
![https://github.com/KwanWooPark97/RL_basic/blob/master/ch6_tf2_ppo/img/ppo_9.png](https://github.com/KwanWooPark97/RL_basic/blob/master/ch6_tf2_ppo/img/ppo_9.png)   
그래서 결국 이러한 식을 결과로 얻는다. 여기서 중요한 사항은 p가 비슷하다는 가정을 어떻게 얻을 수 있냐??  
-> 정책이 비슷하면 p가 비슷하다는 것을 밑에 있는 논문에서 증명했다.  
따라서 우리는 정책이 크게 변하지 않는 만큼만 해당 식을 업데이트 한다.  

![https://github.com/KwanWooPark97/RL_basic/blob/master/ch6_tf2_ppo/img/ppo_10.png](https://github.com/KwanWooPark97/RL_basic/blob/master/ch6_tf2_ppo/img/ppo_10.png)   

PPO에서 사용하는 LOSS의 최종 정리 본  

![https://github.com/KwanWooPark97/RL_basic/blob/master/ch6_tf2_ppo/img/ppo_11.png](https://github.com/KwanWooPark97/RL_basic/blob/master/ch6_tf2_ppo/img/ppo_11.png)  

KL 발산 규제를 통한 최적화 과정이다.  
강화 학습에서는 TRPO와 PPO가 있다. TRPO 또한 성능이 괜찮게 나오지만 구현이 어렵다고 알려져 있다.  
그 이유를 따로 작성해보면 KL 발산의 기본 식은 Hessian Metrix를 필요로 한다. 그렇다는건 Linear가 아닌 quadratic으로 식을 표현한다는 것이다. 이는 계산 복잡성과 차원을 굉~~~장히 크게 만들어 버리는 단점이 있다. 그래서 이걸 구현하기 위한 코드나 계산이 너무 힘들다.  
하지만 PPO는 Linear로 식이 표현 가능하고 뒤에 나오지만 간단하게 cliping으로 이걸 해결해 버린다.  

![https://github.com/KwanWooPark97/RL_basic/blob/master/ch6_tf2_ppo/img/ppo_12.png](https://github.com/KwanWooPark97/RL_basic/blob/master/ch6_tf2_ppo/img/ppo_12.png)  

IS는 그냥 r로 표현해 버렸고 clip를 통해 한번에 크게 변하는 경우를 막아서 가정을 망가뜨리지 않게 한다.  
결국 비제약 최적화 조건으로 상황을 바꿔버렸다.  

아래는 PPO의 의사 코드이다.   
![https://github.com/KwanWooPark97/RL_basic/blob/master/ch6_tf2_ppo/img/ppo_13.png](https://github.com/KwanWooPark97/RL_basic/blob/master/ch6_tf2_ppo/img/ppo_13.png)  
![https://github.com/KwanWooPark97/RL_basic/blob/master/ch6_tf2_ppo/img/ppo_14.png](https://github.com/KwanWooPark97/RL_basic/blob/master/ch6_tf2_ppo/img/ppo_14.png)   

또 다른 PPO의 중요한 계산은 GAE(Generalized Advantage Estimation)이다.  
기존의 어드밴티지 추정 방법에는 두 가지 극단적인 방식이 있다.  

1. 1-step TD(Temporal Difference) 방법: 어드밴티지 함수가 가장 최근의 보상만을 반영하여 추정한다.  
   이는 분산이 적지만, 편향이 클 수 있다.  
2. Monte Carlo 방법: 전체 에피소드가 끝난 후 누적 보상을 기반으로 어드밴티지를 추정한다.
   이는 편향이 없지만, 분산이 클 수 있습니다.
GAE의 핵심 아이디어는 다양한 시점에 대해 어드밴티지를 계산하고, 이를 가중평균하여 어드밴티지 함수를 추정하는 것이다.

![https://github.com/KwanWooPark97/RL_basic/blob/master/ch6_tf2_ppo/img/ppo_15.png](https://github.com/KwanWooPark97/RL_basic/blob/master/ch6_tf2_ppo/img/ppo_15.png)
![https://github.com/KwanWooPark97/RL_basic/blob/master/ch6_tf2_ppo/img/ppo_16.png](https://github.com/KwanWooPark97/RL_basic/blob/master/ch6_tf2_ppo/img/ppo_16.png)
![https://github.com/KwanWooPark97/RL_basic/blob/master/ch6_tf2_ppo/img/ppo_17.png](https://github.com/KwanWooPark97/RL_basic/blob/master/ch6_tf2_ppo/img/ppo_17.png)

λ=1일 때, GAE는 Monte Carlo 방식에 가깝게 작동하여 분산이 크지만 편향이 적은 어드밴티지를 계산한다.  
반면 λ=0일 때는 단순한 1-step TD 추정으로 돌아가, 편향이 크지만 분산이 적은 어드밴티지를 제공한다.  

