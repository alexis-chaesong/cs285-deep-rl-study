# 📐 UC Berkeley CS 285 HW2: 정책 경사(Policy Gradients) 모든 수식 완벽 해설

이 문서는 **CS 285 Assignment 2 (Policy Gradients)** 안내서에 등장하는 모든 수식(식 1 ~ 식 26)을 직관적인 설명과 함께 처음부터 끝까지 차근차근 풀어쓴 해설서입니다.

---

## 1. 기본 개념 및 기호 사전 (Notation)

* $\theta$ **(세타)**: 정책 신경망(Actor)의 가중치(파라미터)입니다. 우리가 학습을 통해 최적화하려는 대상입니다.
* $\phi$ **(파이)**: 가치 신경망(Critic / Baseline)의 가중치입니다.
* $H$ **(지평, Horizon)**: 한 판(에피소드)의 전체 길이(최대 스텝 수)입니다.
* $\tau$ **(타우, 궤적)**: 시점 $0$부터 $H-1$까지 에이전트가 겪은 상태와 행동의 전체 기록입니다.
* $\pi_\theta(a_t | s_t)$ **(정책)**: 상태 $s_t$에서 행동 $a_t$를 선택할 확률입니다.
* $p(s_{t+1} | s_t, a_t)$ **(환경 전이 확률)**: 상태 $s_t$에서 행동 $a_t$를 했을 때 다음 상태 $s_{t+1}$로 넘어갈 물리적 확률입니다.

---

## 2. 정책 경사의 기본 유도 (식 1 ~ 식 8)

### 수식 (1) ~ (3): 강화학습의 목표와 궤적의 확률
$$J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta(\tau)}[r(\tau)] \quad \text{(1)}$$
$$\pi_\theta(\tau) = p(s_0)\pi_\theta(a_0|s_0) \prod_{t=1}^{H-1} p(s_t|s_{t-1}, a_{t-1})\pi_\theta(a_t|s_t) \quad \text{(2)}$$
$$r(\tau) = \sum_{t=0}^{H-1} r(s_t, a_t) \quad \text{(3)}$$

* **의미**: 우리의 목표 $J(\theta)$는 정책 $\pi_\theta$를 따라 움직였을 때 한 판 동안 얻는 **총 보상 $r(\tau)$의 기댓값(평균)**을 최대로 만드는 것입니다.
* **핵심 포인트**: 식 (2)를 보면 궤적이 일어날 확률 $\pi_\theta(\tau)$ 안에는 우리가 조절할 수 있는 **정책($\pi_\theta$)**과, 우리가 알 수 없는 **환경의 물리 법칙($p$)**이 곱셈으로 섞여 있습니다.

### 수식 (4) ~ (6): 로그 미분 트릭 (Log-Derivative Trick)
$$\nabla_\theta J(\theta) = \nabla_\theta \int \pi_\theta(\tau) r(\tau) d\tau \quad \text{(4)}$$
$$= \int \pi_\theta(\tau) \nabla_\theta \log \pi_\theta(\tau) r(\tau) d\tau \quad \text{(5)}$$
$$= \mathbb{E}_{\tau \sim \pi_\theta(\tau)} \left[ \nabla_\theta \log \pi_\theta(\tau) r(\tau) \right] \quad \text{(6)}$$

* **의미**: 미분 항등식 $\nabla_\theta \pi_\theta(\tau) = \pi_\theta(\tau) \nabla_\theta \log \pi_\theta(\tau)$를 이용해 미분 기호($\nabla_\theta$)를 적분 안쪽으로 집어넣고, 다시 기댓값($\mathbb{E}$) 형태로 바꿉니다. 이렇게 해야 실제 시뮬레이션 샘플들의 평균으로 경사를 계산할 수 있습니다.

### 수식 (7) ~ (8): 샘플 평균 근사 (REINFORCE)
$$\nabla_\theta J(\theta) \approx \frac{1}{N} \sum_{i=1}^{N} \nabla_\theta \log \pi_\theta(\tau^i) r(\tau^i) \quad \text{(7)}$$
$$= \frac{1}{N} \sum_{i=1}^{N} \left( \sum_{t=0}^{H-1} \nabla_\theta \log \pi_\theta(a_t^i | s_t^i) \right) \left( \sum_{t=0}^{H-1} r(s_t^i, a_t^i) \right) \quad \text{(8)}$$

* **의미**: 식 (2)에 로그($\log$)를 취하면 곱셈($\prod$)이 덧셈($\sum$)으로 쪼개집니다. 이때 $\theta$와 무관한 환경 전이 확률($\log p$)은 미분하면 $0$이 되어 깔끔하게 사라집니다.
* **결론**: $N$판의 시뮬레이션을 돌린 뒤, **"각 행동의 로그 확률 기울기 $\times$ 그 판의 총 보상"**을 평균 내어 가중치를 업데이트합니다.

---

## 3. 분산을 줄이는 3가지 핵심 트릭 (식 9 ~ 식 14)

식 (8)은 샘플마다 결과가 너무 크게 흔들리는(분산이 큰) 단점이 있습니다. 이를 고치기 위해 세 가지 수학적 트릭을 순서대로 적용합니다.

### ① 인과성: Reward-to-Go (식 9)
$$\nabla_\theta J(\theta) \approx \frac{1}{N} \sum_{i=1}^{N} \sum_{t=0}^{H-1} \nabla_\theta \log \pi_\theta(a_t^i | s_t^i) \left( \sum_{t'=t}^{H-1} r(s_{t'}^i, a_{t'}^i) \right) \quad \text{(9)}$$

* **변화점**: 보상 합계의 시작점이 $t'=0$에서 **$t'=t$**로 바뀌었습니다.
* **의미**: 시점 $t$에 내린 행동은 그 이전($0 \sim t-1$)의 보상에 아무런 영향을 줄 수 없습니다. 과거 보상을 떼어내고 **"이 행동을 한 이후($t$부터 끝까지) 받은 보상"**만 책임지게 하여 불필요한 노이즈(분산)를 줄입니다.

### ② 할인율 적용: Discounting (식 10 & 식 11)
과제 코드(`_calculate_q_vals`)에서 직접 구현하는 두 가지 방식입니다.

* **방식 1 (전체 궤적 기준 절대 할인, 식 10 / 식 25)**:
  $$\nabla_\theta J(\theta) \approx \frac{1}{N} \sum_{i=1}^{N} \sum_{t=0}^{H-1} \nabla_\theta \log \pi_\theta(a_t^i | s_t^i) \left( \sum_{t'=0}^{H-1} \gamma^{t'} r(s_{t'}^i, a_{t'}^i) \right) \quad \text{(10)}$$
  * *문제점*: 게임 시작($t'=0$)을 기준으로 할인율 $\gamma^{t'}$을 곱하므로, 게임 후반부에 일어나는 일들은 보상이 너무 작아져서 에이전트가 후반부 플레이를 대충 하게 됩니다.

* **방식 2 (현재 시점 기준 상대 할인 / Reward-to-go, 식 11 / 식 26 — 실전 표준)**:
  $$\nabla_\theta J(\theta) \approx \frac{1}{N} \sum_{i=1}^{N} \sum_{t=0}^{H-1} \nabla_\theta \log \pi_\theta(a_t^i | s_t^i) \left( \sum_{t'=t}^{H-1} \gamma^{t'-t} r(s_{t'}^i, a_{t'}^i) \right) \quad \text{(11)}$$
  * *의미*: 지금 시점 $t$를 기준으로 얼마나 멀리 떨어져 있는지($t' - t$)에 따라 할인율 $\gamma^{t'-t}$을 곱합니다. 언제든 당장 눈앞의 보상부터 미래의 보상까지 일관된 기준으로 평가합니다.

### ③ 기준선 빼기: Baseline & 가치 함수 (식 12 ~ 식 14)
$$\nabla_\theta J(\theta) = \nabla_\theta \mathbb{E}_{\tau \sim \pi_\theta(\tau)}[r(\tau) - b] \quad \text{(12)}$$
$$\text{왜냐하면 } \nabla_\theta \mathbb{E}_{\tau \sim \pi_\theta(\tau)}[b] = \mathbb{E}_{\tau \sim \pi_\theta(\tau)}[\nabla_\theta \log \pi_\theta(\tau) \cdot b] = 0$$

* **의미**: 보상에서 상수(또는 상태에만 의존하는 값) $b$를 빼주어도 수학적으로 기울기의 평균(기댓값)은 전혀 변하지 않습니다(Unbiased). 대신 점수 숫자의 크기가 작아지면서 분산이 크게 줄어듭니다.

* **신경망 가치 함수를 기준선으로 사용 (식 13 & 식 14)**:
  $$V_\phi^\pi(s_t) \approx \sum_{t'=t}^{H-1} \mathbb{E}_{\pi_\theta} \left[ \gamma^{t'-t} r(s_{t'}, a_{t'}) \middle| s_t \right] \quad \text{(13)}$$
  $$\nabla_\theta J(\theta) \approx \frac{1}{N} \sum_{i=1}^{N} \sum_{t=0}^{H-1} \nabla_\theta \log \pi_\theta(a_t^i | s_t^i) \left( \sum_{t'=t}^{H-1} \gamma^{t'-t} r(s_{t'}^i, a_{t'}^i) - V_\phi^\pi(s_t^i) \right) \quad \text{(14)}$$
  * *의미*: 상태 $s_t$에서 앞으로 평균적으로 얼마를 벌 수 있는지 예측하는 크리틱 신경망 $V_\phi^\pi(s_t)$를 기준선으로 뺍니다. 즉, **"평균 예상 점수($V$)보다 실제로 얼마나 더 잘했는가"**를 기준으로 행동의 확률을 조절합니다.

---

## 4. 일반화된 어드밴티지 추정: GAE (식 15 ~ 식 22)

### 수식 (15) ~ (17): 1스텝 오차($\delta_t$)와 $n$-스텝 어드밴티지
$$A^\pi(s_t, a_t) = Q^\pi(s_t, a_t) - V^\pi(s_t) \quad \text{(15)}$$
$$A^\pi(s_t, a_t) \approx \delta_t = r(s_t, a_t) + \gamma V_\phi^\pi(s_{t+1}) - V_\phi^\pi(s_t) \quad \text{(16)}$$
$$A_n^\pi(s_t, a_t) = \sum_{t'=t}^{t+n-1} \gamma^{t'-t} r(s_{t'}, a_{t'}) + \gamma^n V_\phi^\pi(s_{t+n}) - V_\phi^\pi(s_t) \quad \text{(17)}$$

* **식 (16) ($\delta_t$, 1스텝 TD 오차)**: 끝까지 기다리지 않고 딱 1스텝만 가본 뒤, 나머지는 크리틱의 예측값($V(s_{t+1})$)으로 메웁니다. 분산은 아주 작지만, 크리틱이 틀린 만큼 편향(Bias)이 생깁니다. (판의 마지막 스텝 $H-1$에서는 다음 상태가 없으므로 $\delta_{H-1} = r_{H-1} - V(s_{H-1})$이 됩니다.)
* **식 (17) ($n$-스텝 어드밴티지)**: $n$번만큼은 실제 보상을 더하고, $n$스텝 뒤부터 크리틱 예측값을 쓰는 절충안입니다. $n$이 커질수록 편향은 줄고 분산은 커집니다.

### 수식 (18) ~ (22): 모든 $n$-스텝을 섞는 GAE와 재귀 구현식
$$A_{GAE}^\pi(s_t, a_t) = \frac{1-\lambda}{1-\lambda^{H-t-1}} \sum_{n=1}^{H-t-1} \lambda^{n-1} A_n^\pi(s_t, a_t) \quad \text{(18)}$$
$$\text{정리하면 } \implies A_{GAE}^\pi(s_t, a_t) = \sum_{t'=t}^{H-1} (\gamma\lambda)^{t'-t} \delta_{t'} \quad \text{(21)}$$
$$\text{코드 구현용 재귀식 } \implies A_{GAE}^\pi(s_t, a_t) = \delta_t + \gamma\lambda A_{GAE}^\pi(s_{t+1}, a_{t+1}) \quad \text{(22)}$$

* **의미**: $1$-스텝부터 끝까지의 모든 어드밴티지 추정치에 가중치 $\lambda^{n-1}$을 곱해 지수 가중 평균을 낸 것이 식 (18)입니다.
* **마법의 정리 (식 21 & 22)**: 복잡한 식 (18)을 수학적으로 정리하면 놀랍게도 **"미래의 1스텝 오차($\delta_{t'}$)들에 $(\gamma\lambda)$를 거듭제곱해서 다 더한 것(식 21)"**과 같아집니다.
* 이를 코드로 가장 빠르게 계산하기 위해 뒤에서부터 앞으로 한 칸씩 당겨오며 더하는 공식이 바로 **식 (22)**입니다. (`pg_agent.py`의 `reversed` 루프 코드가 바로 이 식입니다!)

---

## 5. 어드밴티지 정규화 (식 23 ~ 식 24)

$$A^\pi(s_t, a_t) \quad \longrightarrow \quad \frac{A^\pi(s_t, a_t) - \mu}{\sigma + \epsilon} \quad \text{(23), (24)}$$

* **기호 설명**: $\mu$는 현재 배치 내 어드밴티지들의 평균, $\sigma$는 표준편차, $\epsilon$은 $0$으로 나누는 에러를 막기 위한 아주 작은 수($10^{-8}$)입니다.
* **의미**: 구해진 어드밴티지 값들의 크기가 너무 들쭉날쭉하면 신경망 학습률(Learning Rate)을 맞추기가 매우 어렵습니다. 배치 안에서 평균을 $0$, 표준편차를 $1$로 깔끔하게 스케일링해 주어 학습을 안정화하는 실전 필수 테크닉입니다.