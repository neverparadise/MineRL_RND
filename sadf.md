위 그림에서 공간은 weight vector들이 살고 있는 공간이고, 각 선은 각 training data를 잘 분류할 수 있는 경계면을 의미한다. 예츨 들어서 weight 들이 가운데 있는 공들 속에만 있으면 모든 training data를 잘 분류할 수 있는 것이 된다. 이 때 이 공간의 넓이 (weight space)를 통해서 evidence의 lower bound를 구할 수 있다. 

  - Gaussian process 역시 이러한 관점에서 표현될 수 있다. GP prior over function은 '$N(0, K(X, X))$'로 표현된다. 



Evidence and generalization 

 -  이 둘은 Bayesian과 Frequentist를 잘 잇는다. 



PAC-Bayes Analysis

 - 먼저 우리는 분류기가 살고 있는 공간 C 에 대한 prior P와  posterior Q를 정의한다. Posterior는 데이터 dependent하다. 

 - 또한 train과 test는 같은 distribution에서 drawn with iid 되었다고 가정한다. 

 - '$c_D$'는 generalization error이고, '$\hat{c}_S$'는 empirical error이다. 



Bayesian Classifier 

 - 우리가 갖고 있는 bound는 어떤 특정 classifier에 대한 bound가 아니라, 확률적 분류기에 대한 bound이다. 

 - 확률적 분류기는 어떤 test data가 들어왔을 때 classifier 중에서 posterior가 가장 높은 놈을 뽑아서 해당 분류기를 이용해서 분로를 하는 것을 의미한다. 다른 논문에선 Gibbs classifier라고도 하는 것 같다. 

 - 이 부분이 Bayesian과는 조금 다른게, Bayesian은 이러한 classifier의 결과의 posterior mean을 취할 것이기 때문이다. 모 사실 MAP이라고 보면 비슷할 수도 있을 거 같은데..? 

 - 이는 우리가 일반적으로 사용하는 SVM과는 매우 다르다. 

 - 그래서 우리가 추가적으로 필요한 것은 이러한 확률적 분류기와 일반적인 분류기를 연결하는 것이다! 



Definitions for main result Assessing the posterior 

 - 우리가 관심 있는 것은 다음 두 값이다. 

 - 1. $$ Q_D = E_{c~Q}[c_D] $$

 - 2. $$\hat{Q}_S = E{c~Q}[\hat{c}_S]$$

 - 첫 번째 값은 expected generalization error이고, 두 번째 값은 expected empirical error이다. (given posterior) 

 - 이 expectation 때문에 Bayesian이 된다. 

 

 이 때 posterior average에 대한 bound는 다음과 같이 구할 수 있다. 

$$ Pr_{(x, y)}(sgn(E_{c~Q}[c(x)])\ne y) \le 2Q_D $$

즉 이것은 Bayesian이 원하는 classifier의 결과에 대한 posterior mean의 성능에 대한 bound이다. 그리고 이 값은 앞에서 구한 '$Q_D$'에 두 배를 곱한 것과 같다. 왜 두배가 되느냐면 모든 '$x$'에 대해서 '$c~Q$'은 classifier '$c$'의 성능은 최소 0.5일 것이다? 이 부분은 좀 더 생각해 봐야 할 것 같다. 



PAC-Bayes Theorem (★★★★★)

 이 포스팅의 목적인 PAC-Bayes Theorem은 다음과 같다. 

 먼저 어떤 임의의 공간 '$D$'가 있고, classifier의 prior '$P$', confidence '$\delta$'가 있을 때, 

 '$1-\delta$'의 확률로, 데이터 '$S~D^m$', 그리고 classifier의 posterior '$Q$'에 대해서 다음을 만족한다. 

$$KL(\hat{Q}_S || Q_D) \le \frac{  KL(Q||P + ln((m+1)/\delta))  }{m}$$

 위의 식을 사실 보기에 쉽지는 않다. 하나씩 차근차근 봐보자. 

 먼저 왼쪽의 '$KL(\hat{Q}_S || Q_D)$' 은 expected empirical error와 expected generalization error의 분포에 대한 KL-divergence이다. KL-divergence는 measure of closeness이고, VC dimension에서 봤던 generalization error에 대응되는 개념이라고 할 수 있을 것이다. 

 '$  KL(Q||P) = E_{c~Q}[ln \frac{Q(c)}{P(c)}] $' 로 두 분포가 얼마나 비슷한지 나타낸다. 



 오른쪽항을 보자. 이는 두 항으로 이뤄져 있다. 왼쪽의 '$KL(Q||P)$'는 우리가 처음 가정한 function에 대한 prior와 어떤 데이터에 대한 해당 function에 대한 posterior이다. 오른쪽은 당연한 소리인데, 데이터의 dimension에 비례하고 confidence에 반비례하는 term이다. 직관적으로 설명하자면 curse of dimensionality 와도 이어지는 개념이라고 볼 수 있겠다. 결국 우리가 probabilistic classifier를 사용한다면, 해당 classifier의 prior와 posterior를 최대한 비슷하게 할 필요가 있다는 당연한 소리를 수학적으로 증명한 것이라고 할 수 있을 것이다. 



Finite Classes 

 만약 function space가 finite cardinality를 갖는다고 할 때, '$h_1, h_2, ..., h_N$' with prior distribution '$p_1, p_2, ..., p_N$', 그리고 이 posterior가 single function (즉 어떤 데이터가 들어오면 posterior는 singleton)이라고 하자. 이때 generalization bound는 다음과 같다. 

$$KL( \hat{err}(h_i) || err(h_i) ) \le \frac{-log(p_i) + ln((m+1)/\delta)}{m}$$

 

결국 function의 posterior를 prior와 비슷하게 잡는 것이 중요하다.? 



증명 파트... <video 1 50분> 

더보기
 

Other applications

 - Gaussian process

 - PAC-Bayes version of VC dimension 

 - Structured output learning 

 - and so on.. 



Linear classifiers and SVMs

 - Focus in on linear function applications 

 - How the application is made

 - Extensions to learning the prior

 - Some UCI datasets! - non trivial bound! 



PAC-Bayes Bound for SVM

 - 56분.. 어렵다 몬소리지? 





2. PAC-Bayes Analysis: Background and Applications (Video Lecture 2)

두 번째 video lecture이다. 어렵다. 



General perspective

 - Try to connect Bayesian and frequentist 

 - Bayesian: more probabilistic prediction 

 - Frequentist: only iid



Frequentist approach

 - Pioneered in Russia by Vapnik and Chervonenkis

 - Introduced in the west by Valiant under the name of 'probably approximately correct (PAC)' 

 - 결과는 최소 '$1-\delta$'의 확률로 우리가 학습시킨 분류기는 낮은 generalization error를 가질 것이다. 



Bayesian approach

 - Bayesian은 분류기 (혹은 regressor) 에 대한 prior에 대한 가정을 한다. 

 - 그리고 Lik를 정의하고, 가지고 있는 학습 데이터에 대해서 posterior를 정한다. (Bayes rule을 이용해서)

 - 그리고 에러에 대한 분포 (예를 들면 가우시안)를 가정해서, 이를 종합해서 우리의 분류기를 학습시킨다. 

 - 그리고 결과를 expected classification under the posterior에 따라서 학습시킨다. (SVM과는 다르다.)

 - Gaussian process regression은 이를 통해서 분석될 수 있다. 



Version space: evidence 



 - 가운데 있는 원이 version space로 우리가 갖는 데이터를 모두 옳게 분류할 수 있는 분류기의 공간을 의미한다. 





3. Some PAC-Bayesian Theorems (by David Mc Allester)

 - 사람들은 machine learning에서 recipes를 배우는 것이다. 예를 들면 SVM, kernel methods 등등이다. 

 

PAC-Bayes bound 

 - Given prior distribution '$P$' and posterior distribution on '$Q$' on the weight vector (or the classifier), 

 - Generalization error of the Gibbs sampler '$err(Q)$' is bounded by some bound '$B(Q)$' on '$Q$'.

 - 그리고 이 bound는 다음과 같이 정의된다. 

$$B(Q) = \hat{err}(Q) + \sqrt{\hat{err}(Q)c(Q)} + c(Q) $$

$$c(Q) = \frac{2(KL(Q, P)+ln\frac{n+1}{\delta})}{n} $$

 의미를 좀 살펴보면, 결국 bound '$B(Q)$'는 empirical error를 의미하는 '$\hat{err}(Q)$'와 prior '$P$'와 posterior '$Q$'사이의 거리를 뜻하는 KL divergence로 표현이 된다. 



그리고 geometric mean이 arithmetic mean으로 bound가 된다는 성질을 이용하면, '$B(Q)$'는 다음으로 bound된다. 

$$B(Q) \le \frac{3}{2}(\hat{err}(Q)+c(Q)) $$