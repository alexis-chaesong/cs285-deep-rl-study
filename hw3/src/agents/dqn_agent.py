from typing import Sequence, Callable, Tuple, Optional

import torch
from torch import nn

import numpy as np

from infrastructure import pytorch_util as ptu

# DQN의 정의 (한 줄 요약)
# "심층 신경망(Deep Network)을 이용해 모든 행동별 미래 가치 점수($Q$)를 예측하고, 
# 이 점수표를 바탕으로 최적의 행동을 선택하며 경험 창고(리플레이 버퍼)를 통해 학습하는 가치 기반 강화학습 알고리즘"

# 이 코드는 에이전트가 환경 속에서 움직이며 "어떤 행동을 할지 결정(get_action)"하고, 
# 쌓인 경험을 바탕으로 "점수표를 수정(update_critic)"하며, "과녁을 동기화(update_target_critic)"하는 전체 구조를 담고 있음.

class DQNAgent(nn.Module):
    # 에이전트가 태어날 때 필요한 신경망, 옵티마이저, 그리고 DQN 학습에 필요한 하이퍼파라미터들을 세팅하는 장치
    # 의미: nn.Module을 상속받는 DQN 에이전트 클래스를 선언
    # 입력값으로 상태 크기(observation_shape), 행동 개수(num_actions), 신경망을 만들어주는 함수(make_critic), 할인율(discount), 타깃 네트워크 갱신 주기 등이 들어옴 
    def __init__(
        self,
        observation_shape: Sequence[int],
        num_actions: int,
        make_critic: Callable[[Tuple[int, ...], int], nn.Module],
        make_optimizer: Callable[[torch.nn.ParameterList], torch.optim.Optimizer],
        make_lr_schedule: Callable[
            [torch.optim.Optimizer], torch.optim.lr_scheduler._LRScheduler
        ],
        discount: float,
        target_update_period: int,
        use_double_q: bool = False,
        clip_grad_norm: Optional[float] = None,
    ):
        super().__init__()

        self.critic = make_critic(observation_shape, num_actions)               # 실시간으로 학습하며 점수표($Q$)를 수정할 내 신경망을 만듦
        self.target_critic = make_critic(observation_shape, num_actions)        # 앞서 배운 "과녁 고정하기"를 위한 타깃 신경망을 똑같이 하나 더 만듦
        self.critic_optimizer = make_optimizer(self.critic.parameters())        # 내 신경망의 가중치를 업데이트할 Adam 등 옵티마이저를 장착
        self.lr_scheduler = make_lr_schedule(self.critic_optimizer)             # 학습이 진행됨에 따라 학습률(Learning Rate)을 조절해 주는 스케줄러를 세팅

        self.observation_shape = observation_shape
        self.num_actions = num_actions
        self.discount = discount                                # 감마 (γ)
        self.target_update_period = target_update_period        # 타깃 네트워크 갱신 주기 (K스텝)
        self.clip_grad_norm = clip_grad_norm                    # 기울기 폭발 방지용 클리핑 값
        self.use_double_q = use_double_q                        # Double Q 쓸 것인지 여부

        self.critic_loss = nn.MSELoss()                         # 손실 함수로 평균 제곱 오차(MSE) 사용

        self.update_target_critic()                             # 처음 시작할 때 타깃 네트워크에 내 신경망 가중치 복사

        # 의미: 전달받은 설정값들을 에이전트의 속성(self)으로 저장해 두고, 맨 마지막 줄에서 update_target_critic()을 불러 타깃 네트워크를 초기화

    # ε-greedy로 행동 고르기
    # 말로 하면: 동전을 던져서 확률 ε면 아무 행동이나, 아니면 점수가 가장 높은 행동을 고름 
    # 2. 행동 선택: get_action 함수 ($\epsilon$-greedy)
    # 에이전트가 현재 상황을 보고 어떤 행동을 할지 결정할 때 호출하는 함수
    def get_action(self, observation: np.ndarray, epsilon: float = 0.0) -> int:
        """
        Epsilon-greedy action selection (default epsilon=0 for deterministic/greedy policy).
        """
        observation = ptu.from_numpy(np.asarray(observation))[None]
        # 의미: 넘파이 배열 형태의 관측값(observation)을 받아 파이토치 텐서로 변환하고, 신경망에 넣기 위해 배치 차원([None], 맨 앞에 1 추가)을 늘려줌 
        # $\epsilon$-greedy 전략에 따라 확률 $\epsilon$이면 무작위 행동, 아니면 내 신경망(critic)에 상태를 넣어서 나온 점수 중 가장 높은 점수의 행동(argmax)을 골라 action에 담고
        # 마지막 줄에서 파이토치 텐서인 action을 다시 파이썬 숫자(int)로 변환해서 반환(item())

        # TODO(Section 2.4): get the action from the critic using an epsilon-greedy strategy
        if np.random.random() < epsilon:                        # 빈칸 1: 무작위로 고를 확률
            action = torch.randint(self.num_actions, (1,))      # 빈칸 2: 행동이 몇 개인가
        else:
            action = self.critic(observation).argmax(dim=1)     # 빈칸 3: 최신 Q 신경망 이름
        # ENDTODO

        return ptu.to_numpy(action).squeeze(0).item()

    # 3. 크리틱(점수표) 학습: update_critic 함수
    # 리플레이 버퍼에서 뽑아온 경험 묶음을 가지고 목표값($y$)을 만들고 신경망을 훈련하는 핵심 함수
    def update_critic(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        reward: torch.Tensor,
        next_obs: torch.Tensor,
        done: torch.Tensor,
    ) -> dict:
        """Update the DQN critic, and return stats for logging."""
        (batch_size,) = reward.shape
        # 의미: 한 번에 학습할 데이터 묶음의 크기(batch_size)를 보상 텐서의 모양으로부터 추출

        # Compute target values
        with torch.no_grad():
            # TODO(Section 2.4): compute target values
            # 목표값 y 만들기
            next_qa_values = self.target_critic(next_obs)
            # 의미: torch.no_grad() 안에서는 기울기(gradient)를 계산하지 않음
            # 목표값($y$)을 만들 때는 신경망을 학습시키는 것이 아니라 '정답지'를 계산하는 것이므로 가중치가 흔들리면 안 되기 때문

            if self.use_double_q:
                # use_double_q가 켜져 있으면 Double Q 방식(내 신경망이 행동을 고르고, 타깃 신경망이 점수를 쥠)으로 다음 상태의 가치를 구하고, 꺼져 있으면 일반적인 방식로 구함 
                # 최종적으로 벨만 방정식에 따라 목표값 target_values ($y$)를 계산하여 완성
                # Q 예측과 손실
                # TODO(Section 2.5): implement double-Q target action selection
                next_action = self.critic(next_obs).argmax(dim=1)  # 빈칸: 내 신경망이 다음 상태에서 점수가 가장 높은 행동을 고름
            else:
                next_action = next_qa_values.argmax(dim=1)  # 빈칸: 다음 상태에서 점수가 가장 높은 행동을 고름

            next_q_values = next_qa_values.gather(1, next_action.unsqueeze(1)).squeeze(1)  # 빈칸: 다음 상태에서 고른 행동의 점수를 가져옴
            assert next_q_values.shape == (batch_size,), next_q_values.shape

            target_values = reward + self.discount * (1 - done.float()) * next_q_values  # 빈칸: 벨만 방정식에 따라 목표값 계산
            assert target_values.shape == (batch_size,), target_values.shape
            # ENDTODO

        # TODO(Section 2.4): train the critic with the target values
        # 내 신경망(critic)에 현재 상태(obs)를 넣어 실제 내가 고른 행동에 대한 점수(q_values)를 예측
        qa_values = self.critic(obs)
        q_values = qa_values.gather(1, action.unsqueeze(1)).squeeze(1)  # 빈칸: 내가 고른 행동(action)에 대한 점수만 가져옴
        # 위에서 구한 목표값(target_values)과 내 예측값(q_values) 사이의 오차를 MSE 손실 함수(self.critic_loss)를 이용해 계산
        loss = self.critic_loss(q_values, target_values)
        # ENDTODO

        self.critic_optimizer.zero_grad()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad.clip_grad_norm_(                   # 기울기가 너무 커서 학습이 폭발하는 것을 막기 위해 깎아줌(클리핑).
            self.critic.parameters(), self.clip_grad_norm or float("inf")
        )
        self.critic_optimizer.step()

        self.lr_scheduler.step()

        # 의미: 로깅(Wandb 등)을 위해 손실값, Q점수 평균, 목표값 평균 등의 통계 지표를 딕셔너리 형태로 반환
        return {
            "critic_loss": loss.item(),
            "q_values": q_values.mean().item(),
            "target_values": target_values.mean().item(),
            "grad_norm": grad_norm.item(),
        }

    # 4. 타깃 네트워크 갱신: update_target_critic 함수
    # 의미: 내 신경망(critic)의 가중치 상태를 통째로 복사해서, 얼려두었던 과녁인 타깃 신경망(target_critic)에 덮어씀 (과녁 동기화 함수!)
    def update_target_critic(self):
        self.target_critic.load_state_dict(self.critic.state_dict())

    # ③ 호출 + 타깃 네트워크 갱신
    # 5. 에이전트 업데이트 총괄: update 함수
    # 외부 실행 루프에서 매 스텝마다 호출하는 대문 함수
    # 앞에서 본 update_critic을 불러 점수표를 학습시키고 통계값을 받
    # 현재 스텝(step)이 타깃 갱신 주기(target_update_period)의 배수가 되었는지 확인(step % self.target_update_period == 0)하여, 맞다면 update_target_critic()을 불러 과녁을 갱신
    def update(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        reward: torch.Tensor,
        next_obs: torch.Tensor,
        done: torch.Tensor,
        step: int,
    ) -> dict:
        """
        Update the DQN agent, including both the critic and target.
        """
        # TODO(Section 2.4): update the critic, and the target if needed
        critic_stats = self.update_critic(obs, action, reward, next_obs, done)  # 빈칸: 점수표 학습 함수 호출
        # Hint: if step % self.target_update_period == 0: ...
        if step % self.target_update_period == 0:  # 빈칸: 타깃 갱신 주기 확인
            self.update_target_critic()  # 빈칸: 과녁 갱신 함수 호출    
        # ENDTODO

        return critic_stats
