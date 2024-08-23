# SAC(Soft Actor Critic)  

Soft Actor-Critic (SAC)은 강화학습(Reinforcement Learning) 알고리즘 중 하나로, 특히 연속적인 행동 공간을 다루는 데 효과적인 알고리즘입니다.  
SAC는 Off-Policy 방법론에 속하며, 목표는 에이전트가 환경에서 행동할 때 높은 보상을 받는 동시에, 행동의 확률 분포를 최대한 넓게 만들어서 탐색을 장려하는 것입니다.  
이를 위해 SAC는 엔트로피(Entropy)를 최대화하는 정책을 학습합니다. 여기서 엔트로피는 정책의 불확실성이나 다양성을 나타내는 지표로 사용됩니다.  

나는 SAC를 이용해서 해외 저널 논문을 작성하며 진짜 처음부터 구현하는 경험을 해봤다.  
이때 가장 어려웠던 것은 LOSS 코드를 짜는게 제일 헷갈렸다.  
Q 함수 2개를 사용하는데 서로 번갈아 사용하는 경우도 있고 어느 버전은 V함수를 사용해서 업데이트를 진행하는 경우도 있다.  
그래서 notation을 공부하는 과정도 힘들었던 기억이 있다.  
먼저 가장 큰 특징을 살펴보면  
* 탐색과 활용의 균형: 엔트로피를 최대화함으로써 SAC는 탐색과 활용 사이의 균형을 잘 맞춘다.  
* 연속적 행동 공간 처리: SAC는 연속적인 행동 공간을 다룰 때 매우 효과적이다.  
* 안정성: 경험 재사용과 이중 Q-학습 덕분에 안정적인 학습이 가능하다.

이름에 대한 작은 지식으로는 soft가 붙었다는 것은 엔트로피가 들어가서 soft objective 함수를 학습해서 붙은것으로 알고있다.  
이는 정책이 가능한 한 불확실성을 유지하면서 최적의 행동을 학습하도록 만들어준다.  
뒤에 어떠한 효과가 있는지 사진으로 비교하며 다시 설명하겠다.  

강의 자료를 보고 공부할 때 앞부분은 그냥 RL의 기본 개념들을 설명한다.  
notation이나 Q 함수 V 함수의 정의, 벨만 기대 방정식, 벨만 최적 방정식 같은 내용이여서 이부분은 넘어간다.  

![https://github.com/KwanWooPark97/RL_basic/blob/master/ch8_tf2_sac/img/SAC_1.png](https://github.com/KwanWooPark97/RL_basic/blob/master/ch8_tf2_sac/img/SAC_1.png)  

SAC의 중요한 내용인 엔트로피의 기본 개념이다.  
간단하게 목적 함수에 엔트로피 텀을 추가하여 더해준다.  
$\alpha$ 는 온도 함수로 엔트로피의 가중치를 정하는 파라미터이다.  

![https://github.com/KwanWooPark97/RL_basic/blob/master/ch8_tf2_sac/img/SAC_2.png](https://github.com/KwanWooPark97/RL_basic/blob/master/ch8_tf2_sac/img/SAC_2.png)  

엔트로피를 추가하면 q와 v 함수 관계식에 엔트로피 텀이 추가된다.  
위에서 말했듯이 soft가 앞에 붙는 이유는 엔트로피가 들어가있기 때문이다.  

![https://github.com/KwanWooPark97/RL_basic/blob/master/ch8_tf2_sac/img/SAC_3.png](https://github.com/KwanWooPark97/RL_basic/blob/master/ch8_tf2_sac/img/SAC_3.png)  

벨만 기대 방정식에서 똑같이 유도하여 목적함수가 어떻게 변하는지 살펴본다.  
맨밑에 optimal policy 부분에서 j 편미분한 값이 0이라 왜 그런가 생각하며 금방 깨닫고 바보같다고 생각했다.  
그리고 옆에 적분을 직접 풀어보니 밑에 식처럼 나오는 것 까지 확인했다.  

![https://github.com/KwanWooPark97/RL_basic/blob/master/ch8_tf2_sac/img/SAC_4.png](https://github.com/KwanWooPark97/RL_basic/blob/master/ch8_tf2_sac/img/SAC_4.png)  

optimal policy면 j 편미분이 0이므로 deterministic 한 값이 나오니까 앞에서 본것처럼 값이 딱 나오지만  
그게 아니면 greedy 형식으로 진행한다. 이때 softmax 정의를 활용하여 정책을 위에 사진 처럼 나타낼 수 있다.  
그리고 soft q 와 v 함수의 관계 식에서 엔트로피 텀에 있는 정책에 위에서 구한 정책을 그대로 넣으면  
맨 밑과 같이 q 함수가 사라지고 log 형식의 하나의 텀만 남게 된다.  
이 식은 greedy policy를 가지고 있을 때 유도한 식이다.  

![https://github.com/KwanWooPark97/RL_basic/blob/master/ch8_tf2_sac/img/SAC_5.png](https://github.com/KwanWooPark97/RL_basic/blob/master/ch8_tf2_sac/img/SAC_5.png)  

이부분은 KL 발산의 정의를 이용해서 목적 함수를 유도한다.  
확실히 이 방법이 나는 훨씬 이해하기 편했다.  
KL 발산은 두 확률 분포 간의 차이를 측정하는 통계적 방법이다. 특히, 한 확률 분포 P가 다른 확률 분포 Q와 얼마나 다른지를 계산하는 데 사용됩니다.  
이산 확률 분포인 경우:  
$D_{KL}(P \parallel Q) = \sum_{x} P(x) \log \left(\frac{P(x)}{Q(x)}\right)$   
연속 확률 분포인 경우:  
$D_{KL}(P \parallel Q) = \int_{-\infty}^{\infty} P(x) \log \left(\frac{P(x)}{Q(x)}\right) dx$  
이걸 보고 위에 사진에서 KL 발산으로 변환하는 부분을 보자.  
기대값 정의에 의해 연속 확률 분포의 정의에서 P와 Q를 바꾸면 똑같은 꼴을 하고 있으므로 변환할 수 있다.  
KL 발산 오른쪽 텀에 EXP(Q)가 어째서 확률로 정의되냐?? 라고 생각을 했는데 Z를 합쳐서 분수 자체를 하나로 보면 앞에서 정의한 꼴과 비슷하다.  
정확히는 softmax 정의와 똑같은 꼴을 가지므로 이는 확률 분포라고 할 수 있다.  

![https://github.com/KwanWooPark97/RL_basic/blob/master/ch8_tf2_sac/img/SAC_6.png](https://github.com/KwanWooPark97/RL_basic/blob/master/ch8_tf2_sac/img/SAC_6.png)  

그래서 해당 부분은 최종 KL 발산을 사용한 목적 함수를 보여준다.  
그리고 밑에 사진이 굉장히 중요한 내용이다.  
처음에는 무슨 사진인가 고민 했는데 이는 엔트로피의 기능? 성능?을 나타내는 사진이다.  
왼쪽 사진은 엔트로피가 없을 때 상황으로 Q함수가 회색 그래프처럼 존재할 때 정책은 업데이트를 지속할수록  
Q 함수 최대값 부분으로 뾰족한 확률 분포를 가지게 된다.  
물론 이게 나쁜 결과는 아니다 하지만 suboptimal한 Q 값을 보면 최댓값과 엄청난 차이를 내지 않고 엄청 좋지는 않지만 좋은 값을 가지고 있는 것을 볼 수 있다.  
일반적인 부분 argmax를 하므로 저 부분을 무시해버리고 max 부분을 과대평가 해버린다.  
하지만 엔트로피를 추가하면 오른쪽 사진처럼 확률을 조정한다. suboptimal을 신경 쓰게 정책을 만들어버려서 좀 더 유연한? 정책을 만들도록 도와준다.  
이 뒤에는 Lemma가 있다. 필요한 부분만 따로 뽑아서 정리하자.  

![https://github.com/KwanWooPark97/RL_basic/blob/master/ch8_tf2_sac/img/SAC_7.png](https://github.com/KwanWooPark97/RL_basic/blob/master/ch8_tf2_sac/img/SAC_7.png)  

위에서 정리한 KL 발산 식을 이용해 목적함수 J를 재정의한다.  
맨 밑에 식은 어떻게 나왔나 고민했을 때 그 위에 함수에서 Z는 state 함수이다. 하지만 기댓값의 주체는 정책에의한 action이다.  
따라서 state 함수는 상수 취급을 할 수 있고 여기서는 그냥 생략해 버렸다.  
z가 사라지면 log exp 이므로 사라지고 q만 남게된다.  

![https://github.com/KwanWooPark97/RL_basic/blob/master/ch8_tf2_sac/img/SAC_8.png](https://github.com/KwanWooPark97/RL_basic/blob/master/ch8_tf2_sac/img/SAC_8.png)  

당연한 내용이라 그냥 읽어도 문제가 없다.  
새로운 정책에 의한 J가 작은 이유는 업데이트 정책 개선 정리(Policy Improvement Theorem)와 관련이 있다.  
이 정리는 새로운 정책이 이전 정책보다 더 나은(또는 최소한 같거나 더 나은) 행동 가치를 가진다는 것을 보장합니다.  
J를 최소화 하는 학습을 하기 때문에 J가 작아진다.  
그리고 J를 식으로 풀어서 부등호로 나타낸다.  

![https://github.com/KwanWooPark97/RL_basic/blob/master/ch8_tf2_sac/img/SAC_9.png](https://github.com/KwanWooPark97/RL_basic/blob/master/ch8_tf2_sac/img/SAC_9.png)  

여기도 다를거 없이 읽으면서 이해한다.  
부등호의 오른쪽 텀이 -V 함수 정의와 같다. 대입해서 새로운 부등호로 나타낸다.  
그리고 Q 함수 정의를 이용해 새로운 부등호를 만들어서 마지막 식과 같은 결과를 얻을 수 있다.  
이는 학습을 진행하면 항상 새로운 정책의 Q 함수 값이 더 좋은 것을 증명한다.  

![https://github.com/KwanWooPark97/RL_basic/blob/master/ch8_tf2_sac/img/SAC_10.png](https://github.com/KwanWooPark97/RL_basic/blob/master/ch8_tf2_sac/img/SAC_10.png)  

SAC의 최종 정리를 시작한다.  
사용하는 네트워크는 4개이며 V, V_target , Q, actor 이다.  
V함수의 LOSS는 MSE로 정의한다.  

![https://github.com/KwanWooPark97/RL_basic/blob/master/ch8_tf2_sac/img/SAC_11.png](https://github.com/KwanWooPark97/RL_basic/blob/master/ch8_tf2_sac/img/SAC_11.png)  

여기서는 Q 함수의 LOSS를 정의한다. 똑같이 MSE로 사용한다.  
여기서 이해하기 제일 어려운 부분이 왜 Q와 V 함수 2개를 나눠서 사용하냐였다.  
Q 기댓값 만들면 결국 V인데 굳이 나눠서 학습을 또 시켜야되나?? 싶었다.  
이것도 버전에 따라서 Q만 쓰는 SAC가 있고 둘다 쓰는 SAC가 있는데 나는 둘다 쓰는 쪽으로 사용했다.  
식을 보면 Q를 학습시켜 soft V를 예측하도록 도와주고 V_target을 이용해 soft Q를 예측하도록 도와준다.  
GPT에게 물어보니  
V-함수 네트워크를 별도로 두는 것은 학습 효율, 계산 효율, 안정성, 그리고 특정 알고리즘의 요구사항 등을 고려한 결정입니다.  
상황에 따라 V 함수 네트워크를 추가적으로 두는 것이 학습 성능과 효율성을 높이는 데 매우 중요한 역할을 할 수 있습니다.  
라고 한다.  

![https://github.com/KwanWooPark97/RL_basic/blob/master/ch8_tf2_sac/img/SAC_12.png](https://github.com/KwanWooPark97/RL_basic/blob/master/ch8_tf2_sac/img/SAC_12.png)  

재 매개변수화 트릭을 사용해 continuous action space의 경우 actor의 출력을 가우시안 확률 분포로 만들어서 action을 sample한다.  
이 기법의 주요 목적은 미분 가능하지 않은 샘플링 과정을 미분 가능하게 만들어서, 역전파를 통해 신경망을 효율적으로 학습할 수 있도록 하는 것이다.  
원래 샘플링 과정:
𝑧∼𝑁(𝜇,𝜎2)  
이는 평균 μ와 표준편차 σ를 가지는 정규분포에서 샘플링하는 과정이다. 하지만 이 과정은 미분할 수 없습니다.  

재매개변수화:
𝑧=𝜇+𝜎⋅𝜖, 여기서 ϵ∼N(0,1)  
이제 ϵ은 표준 정규분포에서 샘플링된 값으로, μ와 σ는 미분 가능한 파라미터로 남는다.  
이 방식은 z를 μ와 σ에 의존하는 미분 가능한 함수로 표현하여 역전파가 가능해진다.  

actor의 출력이 𝜇,𝜎 이므로 재매개변수화를 통해 미분 가능하도록 바꿔준다.  

![https://github.com/KwanWooPark97/RL_basic/blob/master/ch8_tf2_sac/img/SAC_13.png](https://github.com/KwanWooPark97/RL_basic/blob/master/ch8_tf2_sac/img/SAC_13.png)  

내 연구실에서 일하면 무조건 이해할 수 밖에 없는 SAC 구조이다. 이 구조도를 먼저 보고 연구를 진행했다.  
