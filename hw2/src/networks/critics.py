import itertools
from torch import nn
from torch.nn import functional as F
from torch import optim

import numpy as np
import torch
from torch import distributions

from infrastructure import pytorch_util as ptu

# 7장에서 배운 액터-크리틱 구조 중 크리틱(Critic, 가치 네트워크)을 구현하는 코드
# 크리틱의 역할은 간단합니다. "현재 상태($s$)를 입력으로 받아, 앞으로 얼마만큼의 보상을 받을 수 있을지 그 가치($V(s)$)를 예측"하는 회귀(Regression) 모델임.
# 교재 43장의 몬테카를로/부트스트랩 타깃을 이용해 네트워크를 학습시키는 핵심 코드가 여기에 들어 있음.
class ValueCritic(nn.Module):
    """Value network, which takes an observation and outputs a value for that observation."""

    # 1. 크리틱 네트워크 초기화: __init__
    # 크리틱이 태어날 때 어떤 신경망 구조와 최적화 도구를 가질지 세팅
    def __init__(
        self,
        ob_dim: int,                # 상태(Observation) 차원 수
        n_layers: int,              # 신경망 층(layer) 개수
        layer_size: int,            # 각 층의 노드 수
        learning_rate: float,       # 크리틱 학습률
    ):
        super().__init__()
        # 1. 다층 퍼셉트론(MLP) 신경망 생성
        self.network = ptu.build_mlp(
            input_size=ob_dim,
            output_size=1,          # 가치(Value)는 하나의 숫자(스칼라)이므로 출력 크기가 1
            n_layers=n_layers,
            size=layer_size,
        ).to(ptu.device)

        # 2. Adam 옵티마이저 생성 (가중치를 업데이트할 도구)
        self.optimizer = optim.Adam(
            self.network.parameters(),
            learning_rate,
        )

        # 핵심 의미: 상태 벡터를 받아서 스칼라 값 하나(가치 $V$)를 뱉어내는 전형적인 회귀 신경망과 Adam 옵티마이저를 세팅

    # 2. 순전파: forward 함수
    # 관측값을 신경망에 통과시켜 예측 가치 값을 뽑아내는 함수
    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        # TODO: implement the forward pass of the critic network
        return self.network(obs).squeeze(-1)
        # 구현 코드: return self.network(obs)
        # 핵심 의미: 입력받은 상태 텐서(obs)를 신경망에 통과시켜 예측된 가치 $\hat{V}(s)$를 반환

        # .squeeze(-1)이 왜 필요한가 (중요!)
        # 신경망 출력이 1개라서 모양이 (배치, 1). 그런데 정답 q_values는 (배치,)
        # 신경망 출력	(3, 1)	[[2.0], [1.5], [1.0]]
        # q_values	(3,)	[3.0, 2.0, 1.0]
        # 이 둘을 그냥 빼면 에러 없이 파이썬이 알아서 모양을 맞춰서(브로드캐스팅) (3, 3) 표를 만듦.
        # 9개 칸에서 모든 예측과 모든 정답을 서로 빼 버리는데 그러면 손실 값이 엉뚱하게 계산되는데, 에러가 안 나서 알아채기 어려움 
        # .squeeze(-1)은 마지막의 크기 1인 차원을 없애서 (3, 1) → (3,)로 맞춰 줌  "에러가 안 나는 실수가 더 위험하다"는 HW2 run.py 인자 순서 때와 같은 교훈

    # 3. 크리틱 학습(업데이트): update 함수
    # 7장에서 크리틱을 학습시킬 때 쓰는 회귀 손실 함수(Loss Function)와 역전파 과정이 담기는 핵심 함수
    # 교재 43장의 손실 함수 $\mathcal{L}(\phi) = \frac{1}{2} \sum_i \Vert{}\hat{V}_\phi(s_i) - y_i\Vert{}^2$를 그대로 코드로 옮기는 작업
    def update(self, obs: np.ndarray, q_values: np.ndarray) -> dict:
        # 1. 넘파이 배열을 파이토치 텐서로 변환 (GPU 등 장치로 이동)
        obs = ptu.from_numpy(obs)
        q_values = ptu.from_numpy(q_values)

        # TODO: compute the loss using the observations and q_values

        # 3. 평균 제곱 오차(MSE Loss)를 이용해 손실 계산 (실제 정답인 q_values와 예측값 비교)
        loss = ((self(obs) - q_values) ** 2).mean()

        # 4. 표준 파이토치 최적화 3단계 수행
        # TODO: perform an optimizer step
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {
            "Baseline Loss": loss.item(),       # 로그 기록용 손실값 반환
        }