---
layout: default
title: Boosting Methods
parent: Machine Learning
nav_order: 1
---

부스팅은 여러 약한 분류기(weak classifier)들을 합쳐서 강력한 모델을 만드는 학습방법이다. 
가장 대표적인 부스팅 알고리즘으로는 **AdaBoost**가 있다.

## AdaBoost

-1과 1의 값을 갖는 response variable $ Y\in \{ -1, 1\} $가 있고, $X$를 predictor variable이라 하자.
분류기를 $G$라 했을 때, training sample의 error rate는 다음과 같이 구할 수 있다.
$$
\bar{\text {err}} =\frac{1}{N}\sum\limits_{i=1}^{N}\mathbb I(y_{i}\neq G(x_i))
$$
그리고 error rate의 기댓값은 $\mathbb E[\mathbb I(Y\neq G(X))]$ 이다.

부스팅의 목적은 약한 분류 알고리즘(weak classification algorithm)을 순차적으로 엮는데 있다.

약한 분류기를 $G_{m}(x), \; m=1,2,\ldots,M$ 이라 하자. 
첫 번째 분류기 $G_{1}(x)$는 training sample로 학습하고 학습된 정보를 바탕으로 두 번째 분류기 $G_{2}(x)$부터는 weighted sample로 학습하게 된다. 이 분류기를 $M$개 합쳤을 때 최종적으로 학습된 분류기 $G(x)$가 나오게 된다.
$$
G(x) = \text{sign} \left( \sum\limits_{m=1}^{M}\alpha_{m}G_{m}(x)  \right )
$$
여기서 $\alpha_{1}, \alpha_{2}, \ldots ,\alpha_{M}$ 은 부스팅 알고리즘으로 계산된다. 

각각의 부스팅 단계를 거치면서 $N$개의 데이터에 대한 가중치 $w_{1}, w_{2}, \ldots , w_{N}$ 가 바뀌게 된다. 
당연히 처음의 가중치는 $w_{i}= 1/N$ 이다. 

$m$번째 단계에서 잘못 분류된 데이터 대한 가중치를 증가시키고 잘 분류된 데이터의 가중치는 감소시키는 방향으로 학습하게 된다. 즉, 이전 단계의 분류기 $G_{m-1}(x)$ 에서 에러가 발생한 데이터에 가중치를 더 주어서 학습하는 것이다. 

### AdaBoost.M1 Algorithm

1) 가중치 $w_{i}= 1/M, \; i=1,2,\ldots, N$ 를 할당한다.
2) $m=1$부터 $M$까지 다음을 반복한다.
	1) 가중치 $w_{i}$를 적용시킨 데이터로 분류기 $G_{m}(x)$을 학습시킨다.
	2) error를 계산한다. $$\text {err}_{m} = \frac{\sum\limits_{i=1}^{N}w_{i}\mathbb I(y_{i}\neq G_{m}(x_{i}))}{\sum\limits _{i=1}^{N}w_i}$$
	3) $\alpha_{m}=\log ((1-\text{err}_{m}/\text{err}_m))$ 을 계산한다.
	4) 가중치를 업데이트한다. $$w_{i} \leftarrow w_{i}\cdot \exp[\alpha_{m}\cdot \mathbb I (y_{i}\neq G_{m}(x_{i}))], \; i=1,2,\ldots,N$$



