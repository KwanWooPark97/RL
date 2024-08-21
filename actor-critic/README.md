off-policy의 대표인 DQN을 저번에 다뤘으니 이번엔 on-policy의 시작인 A2C를 공부한다.

 

내가 공부한 알고리즘들은 대부분 A2C처럼 actor와 critic으로 나눠서 학습시키는 것이 대부분이었다. 그만큼 이 논문이

강화 학습에 큰 영향을 끼쳤다고 생각한다.  
![https://github.com/KwanWooPark97/RL_basic/blob/master/actor-critic/img/a2c_1.png](https://github.com/KwanWooPark97/RL_basic/blob/master/actor-critic/img/a2c_1.png)  
 


먼저 objective 함수를 바꿔본다. 궤적을 0:t와 t+1:T 까지로 나눠서 식을 새로 만든다. 이렇게 보면 t+1:T까지에 대한 식은 Q 함수의 정의와 같아진다.

 ![https://github.com/KwanWooPark97/RL_basic/blob/master/actor-critic/img/a2c_2.png](https://github.com/KwanWooPark97/RL_basic/blob/master/actor-critic/img/a2c_2.png)  


Q 함수를 사용하여 목적 함수를 다시 정의하면 위 사진과 같다. 처음봤을 때는 식이 굉장히 어려워 보였는데 이제는 

기댓값의 정의와 Q함수의 정의만 이용해서 식을 전개했다는 것을 이해할 수 있다. 

![https://github.com/KwanWooPark97/RL_basic/blob/master/actor-critic/img/a2c_3.png](https://github.com/KwanWooPark97/RL_basic/blob/master/actor-critic/img/a2c_3.png)  
위 식의 장점은 더이상 에피소드가 끝날 때까지 기다리지 않아도 된다는 것이다. 왜냐하면 Q 함수는 미래의 보상 기댓값을 estimate 하기 때문에 t 시점에서 Q 함수를 구하면 미래의 값을 어느 정도 예측해서 계산 가능하기 때문이다.

 
![https://github.com/KwanWooPark97/RL_basic/blob/master/actor-critic/img/a2c_4.png](https://github.com/KwanWooPark97/RL_basic/blob/master/actor-critic/img/a2c_4.png)  

하지만 문제는 Variance가 크다는 것이다. t 시점에서 Q 함수로 미래 보상을 예측한다고는 하지만 미래의 일은 알수없다. 수많은 가능성이 있고 수많은 경우의 수가 존재하기 때문에 Variance가 커질 수밖에 없다. 따라서 Variance를 줄이기 위해 baseline을 이용한다. 

![https://github.com/KwanWooPark97/RL_basic/blob/master/actor-critic/img/a2c_5.png](https://github.com/KwanWooPark97/RL_basic/blob/master/actor-critic/img/a2c_5.png)  
baseline을 사용해도 gradient 값은 변하지 않는다는 것을 보여준다. A2C에서는 baseline으로 V 함수를 사용했는데 나는 이부분을 목적 함수는 θ에 대한 함수이고 Q 또한 θ에 대한 함수인데 V 함수는 θ가 아닌 state에 대한 함수이므로 θ에 대해서 미분을 하면 0이 나오기 때문에 baseline으로 사용 가능하다고 생각한다.  
내가 이해한 게 맞는지 아닌지는 정확하지 않다.

![https://github.com/KwanWooPark97/RL_basic/blob/master/actor-critic/img/a2c_6.png](https://github.com/KwanWooPark97/RL_basic/blob/master/actor-critic/img/a2c_6.png)  


A2C 논문에서는 baseline을 V 함수로 사용한 경우 새로운 함수를 정의 했다. Advantage 함수라 불리며 A 함수라고 사용한다. 

A 함수의 뜻을 살펴보면 Q 함수는 현재 state에서 특정 action에 대해 보상의 기댓값을 계산하고 V는 현재 state에서의 보상의 기댓값을 계산한다. 그리고 V는 Q함수의 기댓값을 의미한다.

다시 말해서 Q-V가 의미하는 것은 현재 Q 함수의 값이 평균 값보다 얼마나 좋은지를 판단해준다. A 값이 음수 값을 가지면 해당 action은 좋지 않다는 것을 뜻하고 양수 값을 가지면 평균보다는 좋은 action을 의미한다.

그리고 여기서 actor와 critic에 대해 나누기 시작한다. A 를 사용하기 위해서는 3개의 NN을 사용할지 2개의 NN을 사용할지 정할 수 있다. 보통 2개의 NN을 사용해서 policy를 estimate 하는 ACTOR와 V 함숫값을 estimate 하는 CRITIC을 사용한다.
![https://github.com/KwanWooPark97/RL_basic/blob/master/actor-critic/img/a2c_7.png](https://github.com/KwanWooPark97/RL_basic/blob/master/actor-critic/img/a2c_7.png)  

강화 학습은 데이터를 sampling 하기 때문에 위와 같이 식을 바꿀 수 있다. 그리고 Q 함수는 별반 방정식을 사용하여 V 함수에 대해 바꿀 수 있으므로 A 함수를 계산할 수 있다.
![https://github.com/KwanWooPark97/RL_basic/blob/master/actor-critic/img/a2c_8.png](https://github.com/KwanWooPark97/RL_basic/blob/master/actor-critic/img/a2c_8.png)  

위에서 정리한 식들을 사용하여 loss 함수를 구해보면 Critic은 A 함수를 추정하기 위한 loss를 정의했고 Actor는 A함수를 사용해 목적 함수를 최대화하는 방향으로 loss를 정의한다.
![https://github.com/KwanWooPark97/RL_basic/blob/master/actor-critic/img/a2c_9.png](https://github.com/KwanWooPark97/RL_basic/blob/master/actor-critic/img/a2c_9.png)  

마지막으로 action이 discrete 하냐 continuos 하냐에 따라 Actor의 출력이 달라진다는 것을 설명한다.

discrete action의 경우 큰 문제없이 마지막 출력층에서 action_dim을 출력하면 문제가 없다. 하지만 continuous action의 경우 Actor의 출력층은 mean과 variance를 출력하고 이를 Gaussian distribution에 사용하여 action을 확률에 맞게 가져오게 된다. 
