from typing import Optional, Sequence
import numpy as np
import torch

from networks.critics import ValueCritic
from networks.policies import MLPPolicyPG
from infrastructure import pytorch_util as ptu
from torch import nn

class PGAgent(nn.Module):

    # 1. 클래스 초기화: __init__ 함수
    # 에이전트가 세상에 태어날 때 어떤 신경망과 설정값을 가질지 세팅하는 단계
    def __init__(
        self,
        ob_dim: int,                # 상태(Observation) 차원 수
        ac_dim: int,                # 행동(Action) 차원 수
        discrete: bool,             # 이산 행동인지 연속 행동인지 여부
        n_layers: int,              # 신경망 층(layer) 개수
        layer_size: int,            # 각 층의 노드 수
        gamma: float,               # 할인율 (gamma, 7장)
        learning_rate: float,       # 액터 학습률
        use_baseline: bool,         # 기준선(Baseline / Critic)을 쓸 것인가? (6장, 7장)
        use_reward_to_go: bool,     # Reward-to-go를 쓸 것인가? (6장 트릭 ①)
        baseline_learning_rate: Optional[float],    # 크리틱 학습률
        baseline_gradient_steps: Optional[int],     # 크리틱을 몇 번 업데이트할 것인가
        gae_lambda: Optional[float],        # GAE 파라미터 람다 (7장)
        normalize_advantages: bool,         # 어드밴티지 정규화 여부
    ):
        super().__init__()

        # create the actor (policy) network
        # 1. 액터(정책) 신경망 생성: 상태를 받아 행동 확률분포를 출력함 (6장)
        self.actor = MLPPolicyPG(
            ac_dim, ob_dim, discrete, n_layers, layer_size, learning_rate
        )

        # 2. 크리틱(기준선) 신경망 생성: 상태를 받아 가치 V(s)를 예측함 (7장)
        # create the critic (baseline) network, if needed
        if use_baseline:
            self.critic = ValueCritic(
                ob_dim, n_layers, layer_size, baseline_learning_rate
            )
            self.baseline_gradient_steps = baseline_gradient_steps
        else:
            self.critic = None # 기준선을 안 쓰면 크리틱도 없음 (순수 REINFORCE)

        # other agent parameters
        # 3. 기타 하이퍼파라미터 저장
        self.gamma = gamma
        self.use_reward_to_go = use_reward_to_go
        self.gae_lambda = gae_lambda
        self.normalize_advantages = normalize_advantages

    # 2. 메인 학습 주기: update 함수
    # 로봇이 환경을 돌아다니며 모은 데이터(obs, actions, rewards, terminals)를 가지고 정책을 한 걸음 개선(Update)하는 핵심 함수
    # 6장 40쪽의 "가짜 손실" 코드가 동작하는 바탕이 바로 이 함수
    def update(
        self,
        obs: Sequence[np.ndarray],
        actions: Sequence[np.ndarray],
        rewards: Sequence[np.ndarray],
        terminals: Sequence[np.ndarray],
    ) -> dict:
        # 입력값: 여러 판(궤적) 동안 수집한 관측값(obs), 행동(actions), 보상(rewards), 에피소드 종료 여부(terminals)의 리스트
        """The train step for PG involves updating its actor using the given observations/actions and the calculated
        qvals/advantages that come from the seen rewards.

        Each input is a list of NumPy arrays, where each array corresponds to a single trajectory. The batch size is the
        total number of samples across all trajectories (i.e. the sum of the lengths of all the arrays).
        """

        # Step 1: 각 (s_t, a_t) 지점에서의 Q값(또는 Return)을 계산함
        # 의미: 뒤에서 살펴볼 _calculate_q_vals 함수를 호출해, 각 스텝마다 "이후로 얼마의 보상을 얻을 수 있는지" 시뮬레이션 결과 기반으로 계산
        # step 1: calculate Q values of each (s_t, a_t) point, using rewards (r_0, ..., r_t, ..., r_T)
        q_values: Sequence[np.ndarray] = self._calculate_q_vals(rewards)

        # TODO: flatten the lists of arrays into single arrays, so that the rest of the code can be written in a vectorized
        # way. obs, actions, rewards, terminals, and q_values should all be arrays with a leading dimension of `batch_size`
        # beyond this point.
        # 리스트로 쪼개져 있던 여러 판의 데이터를 하나의 거대한 1차원 배열로 이어붙임 (벡터화)
        # 의미: 데이터가 판별로 쪼개져 있으면 신경망 학습이 느리고 번거로움
        # np.concatenate를 통해 모든 스텝의 데이터를 하나로 합쳐서(Flatten) 파이토치 신경망에 통째로 넣을 준비를 함 
        obs = np.concatenate(obs)
        actions = np.concatenate(actions)
        rewards = np.concatenate(rewards)
        terminals = np.concatenate(terminals)
        q_values = np.concatenate(q_values)

        # step 2: calculate advantages from Q values
        # Step 2: Q값에서 기준선을 빼서 '어드밴티지(Advantage)'를 계산함
        # 의미: "이 행동이 평균보다 얼마나 더 좋았는가?"를 나타내는 어드밴티지를 구함. (6장의 기준선 트릭 및 7장의 액터-크리틱/GAE가 여기서 작동).
        advantages: np.ndarray = self._estimate_advantage(
            obs, rewards, q_values, terminals
        )

        # step 3: use all datapoints (s_t, a_t, adv_t) to update the PG actor/policy
        # Step 3: 구한 어드밴티지를 가중치로 써서 액터(정책) 신경망을 업데이트함 (6장 정책 경사)
        # 의미: 정책 경사의 핵심 공식인 ∇log π * Advantage를 실제로 계산하여 역전파(backward())를 수행하고 정책의 가중치를 고침 
        # TODO: update the PG actor/policy network once using the advantages
        info: dict = self.actor.update(obs, actions, advantages)

        # step 4: if needed, use all datapoints (s_t, a_t, q_t) to update the PG critic/baseline
        # Step 4: 크리틱(기준선)이 존재한다면, 예측 가치 V와 실제 값의 오차를 줄이도록 크리틱도 학습시킴 (7장)
        if self.critic is not None:
            # TODO: perform `self.baseline_gradient_steps` updates to the critic/baseline network
            # (여기에 크리틱 신경망을 회귀(Regression) 방식으로 학습시키는 코드가 들어감)
            # 의미: 7장에서 크리틱($V$)이 정확해야 어드밴티지도 정확해집니다. 따라서 크리틱 자신의 예측 오차(Loss)를 줄이는 학습을 함께 수행
            critic_info = None
            for _ in range(self.baseline_gradient_steps):
                critic_info = self.critic.update(obs, q_values)

            info.update(critic_info)

        return info

    # 3. 보상 계산 도우미 함수들
    # 6장 트릭 ①인 인과성(Reward-to-go)의 유무에 따라 보상을 어떻게 누적할지 결정하는 함수들
    # 의미 (기본 방식): 한 판 전체($0$부터 $T$까지)의 보상에 할인율($\gamma$)을 곱해가며 다 더한 총합을 구함 
    def _discounted_return(self, rewards: Sequence[float]) -> Sequence[float]:
        """
        Helper function which takes a list of rewards {r_0, r_1, ..., r_t', ... r_T} and returns
        a list where each index t contains sum_{t'=0}^T gamma^t' r_{t'}

        Note that all entries of the output list should be the exact same because each sum is from 0 to T (and doesn't
        involve t)!
        """
        total = 0
        for t in range(len(rewards)):
            total += rewards[t] * (self.gamma ** t)
        return [total] * len(rewards)
        # 특징: 함수 마지막 줄을 보면 [total] * len(rewards)로, 모든 스텝에 똑같은 총합 점수가 복사되어 들어감. (6장에서 비판했던 "시작이 좋았으니 끝에 망했어도 다 잘했다" 식의 거친 방식)

    # 의미 (Reward-to-go, 6장 트릭 ①): "과거의 보상은 내버려두고, 지금 이 순간($t$)부터 끝($T$)까지 받을 보상만 누적한다"는 철학을 코드로 구현한 것
    # 코드 동작: 리스트의 맨 뒤(끝 스텩)부터 거꾸로 역순(reversed)으로 오면서 현재 보상 + (gamma * 이전까지 누적된 보상)을 차곡차곡 쌓음.
    # 이러면 각 스텩마다 '그 이후에 받은 보상들의 합'이 정확하게 계산
    def _discounted_reward_to_go(self, rewards: Sequence[float]) -> Sequence[float]:
        """
        Helper function which takes a list of rewards {r_0, r_1, ..., r_t', ... r_T} and returns a list where the entry
        in each index t is sum_{t'=t}^T gamma^(t'-t) * r_{t'}.
        """
        out = [0] * len(rewards)
        total = 0
        for t in reversed(range(len(rewards))):
            total = rewards[t] + self.gamma * total
            out[t] = total

        return out

    # 의미: 설정값(use_reward_to_go)에 따라 위에서 설명한 두 방식 중 어떤 것으로 Q값을 추정할지 갈아끼우는 분기문
    def _calculate_q_vals(self, rewards: Sequence[np.ndarray]) -> Sequence[np.ndarray]:
        """Monte Carlo estimation of the Q function."""

        if not self.use_reward_to_go:
            # Case 1: in trajectory-based PG, we ignore the timestep and instead use the discounted return for the entire
            # trajectory at each point.
            # In other words: Q(s_t, a_t) = sum_{t'=0}^T gamma^t' r_{t'}
            # TODO: use the helper function self._discounted_return to calculate the Q-values
            q_values = [self._discounted_return(r) for r in rewards]
        else:
            # Case 2: in reward-to-go PG, we only use the rewards after timestep t to estimate the Q-value for (s_t, a_t).
            # In other words: Q(s_t, a_t) = sum_{t'=t}^T gamma^(t'-t) * r_{t'}
            # TODO: use the helper function self._discounted_reward_to_go to calculate the Q-values
            q_values = [self._discounted_reward_to_go(r) for r in rewards]

        return q_values

    # 4. 어드밴티지 계산의 핵심: _estimate_advantage 함수
    # 6장의 기준선과 7장의 액터-크리틱/GAE 공식을 실제로 구현하는 가장 복잡하면서도 중요한 함수
    def _estimate_advantage(
        self,
        obs: np.ndarray,
        rewards: np.ndarray,
        q_values: np.ndarray,
        terminals: np.ndarray,
    ) -> np.ndarray:
        """Computes advantages by (possibly) subtracting a value baseline from the estimated Q-values.

        Operates on flat 1D NumPy arrays.
        """
        if self.critic is None:
            # TODO: if no baseline, then what are the advantages?
            # Case 1: 기준선이 아예 없으면, Q값이 곧 어드밴티지임 (순수 REINFORCE)
            advantages = q_values
        else:
            # Case 2: 크리틱(기준선)이 존재함
            # (여기에 크리틱 신경망을 상태 obs에 통과시켜 가치 V(s)를 구하는 코드가 들어감)
            # TODO: run the critic and use it as a baseline
            values = ptu.to_numpy(self.critic(ptu.from_numpy(obs)))
            assert values.shape == q_values.shape

            if self.gae_lambda is None:
                # TODO: if using a baseline, but not GAE, what are the advantages?
                # Case 2-a: GAE를 안 쓴다면, 단순하게 Q값에서 크리틱 가치를 뺌 (A = Q - V)
                advantages = q_values - values
            else:
                # TODO: implement GAE
                # Case 2-b: GAE(Generalized Advantage Estimation)를 사용함 (7장)
                batch_size = obs.shape[0]

                # HINT: append a dummy T+1 value for simpler recursive calculation
                # 마지막 스텝 처리를 편하게 하기 위해 더미 값 0을 뒤에 이어붙임
                values = np.append(values, [0])
                advantages = np.zeros(batch_size + 1)

                for i in reversed(range(batch_size)):
                    # TODO: recursively compute advantage estimates starting from timestep T.
                    # HINT: use terminals to handle edge cases. terminals[i] is 1 if the state is the last in its
                    # trajectory, and 0 otherwise.
                    # 판이 계속되면 1, 판의 마지막 칸이면 0
                    not_done = 1 - terminals[i]
                    # ① δ: 한 걸음 뒤 "생각보다 얼마나 좋았나"
                    delta = rewards[i] + self.gamma * values[i + 1] * not_done - values[i]
                    # TODO: 7장에서 배운 1스텝 오차(δ)와 람다를 이용해 
                    # 미래의 어드밴티지들을 재귀적으로 엮어서 계산하는 GAE 루프가 들어감
                    # ② A: 뒤에서부터 쌓기
                    advantages[i] = delta + self.gamma * self.gae_lambda * advantages[i + 1] * not_done

                # remove dummy advantage
                advantages = advantages[:-1] # 더미 값 떼어내기

                # 의미:기준선이 없으면 그냥 $Q$값을 그대로 어드밴티지로 씀 
                # 기준선이 있으면 $Q - V$를 하거나, 7장의 핵심인 GAE 알고리즘을 돌려서 편향과 분산이 가장 알맞게 조율된 정교한 어드밴티지를 계산

        # TODO: normalize the advantages to have a mean of zero and a standard deviation of one within the batch
        # 정규화: 어드밴티지 값들의 평균을 0, 표준편차를 1로 만들어줌
        # 의미: 신경망 학습을 안정적으로 만들기 위해 구해진 어드밴티지 벡터를 표준화(Normalization)함. 
        # 값이 너무 튀거나 작으면 학습이 불안정해지는데, 이를 방지하는 실전적인 필수 팁
        if self.normalize_advantages:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        return advantages
