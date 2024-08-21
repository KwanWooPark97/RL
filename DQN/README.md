이번에는 강화학습의 시작을 알리는 DQN을 공부해보자. 

처음 DQN은 알파고에 사용한 알고리즘으로 알파고가 엄청나게 유명해지면서 사용된 기술이 무엇인지 알아보고자 하는 사람들이 많아진 것이 강화학습에 사람들이 관심을 가지기 시작한게 아닌가 추측해본다.

DQN부터는 V 함수와 Q 함수를 대부분 딥러닝 방법을 사용하여 estimate 하는 방향으로 강화학습이 연구된다. 심층 강화학습의 시작을 알리는 알고리즘인 만큼 강화학습 공부를 시작한다 하면 모든 책이나 강의들은 기본적인 이론을 다루고나서 DQN으로 본격적인 강화학습에 대해서 시작한다.

![https://github.com/KwanWooPark97/RL_basic/blob/master/DQN/img/dqn_1.png](https://github.com/KwanWooPark97/RL_basic/blob/master/DQN/img/dqn_1.png)  

앞에서 배웠던 Q 함수를 이제는 뉴럴 네트워크를 사용하여 표현한다. 앞으로 네트워크를 통해 estimate한 Q함수에는 θ가 붙게된다.

뉴럴 네트워크를 사용한다는 것은 가중치를 학습하기 위한 Loss 또는 Objective 함수가 필요하다는 조건이 생긴다. 따라서 DQN에서는 Mean Squared Error를 사용했는데 여기서 target 값을 어떻게 하는지 보면 TD를 사용하여 target값을 구했다. 

![https://github.com/KwanWooPark97/RL_basic/blob/master/DQN/img/dqn_2.png](https://github.com/KwanWooPark97/RL_basic/blob/master/DQN/img/dqn_2.png)  
Q 네트워크의 구조는 위와 같다. 입력으로는 state가 들어가고 출력으로는 Q 함수가 나온다. 

![https://github.com/KwanWooPark97/RL_basic/blob/master/DQN/img/dqn_3.png](https://github.com/KwanWooPark97/RL_basic/blob/master/DQN/img/dqn_3.png)  
의사 결정 코드를 보면 처음에는 랜덤 가중치를 가진 Q 네트워크를 만들어주고 앱실론 탐욕 확률에 따라서 action을 만들어 준다. 앱실론이 아니라면 Q 함수의 argmax를 취한 action을 구해준다. 

구한 action을 취한후 next_state와 reward를 구해준다. 그리고 off-policy인 만큼 experience replay buffer에 

state,action,reward,next_state를 저장해주고 나중에 학습할때 가져온다. 

buffer에서 batch_size 만큼 데이터를 가져와서 네트워크 학습을 진행해준다. DQN은 Q 네트워크만 학습시키는 간단한 알고리즘이기 때문에 의사결정 코드만 읽어보면 어떤식으로 진행해야할지 감이 잡힌다.
