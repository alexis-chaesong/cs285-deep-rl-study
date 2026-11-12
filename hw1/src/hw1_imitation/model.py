"""Model definitions for Push-T imitation policies."""

# ============================================================================
# model.py : 정책(신경망) 담당 파일
# ----------------------------------------------------------------------------
# 역할 : "상태를 보고 행동 묶음을 내는" 정책 신경망을 정의한다.
# 입력 : state (B, 5)            B = 배치 크기 (학습 128, 평가 1)
# 처리 : MLP로 행동 묶음 예측 (MSE 정책) / 노이즈를 조금씩 이동시켜 생성 (flow 정책)
# 출력 : action_chunk (B, K, 2)   K = chunk_size (기본 8)
#
# 구조
#   BasePolicy         : 모든 정책이 지켜야 할 규칙 (함수 2개를 반드시 구현)
#   MSEPolicy          : HW1 1부. 한 번에 행동 묶음을 예측, MSE 손실
#   FlowMatchingPolicy : HW1 2부. flow matching (내일 구현)
#   build_policy       : 이름("mse"/"flow")을 받아 해당 정책 객체를 만들어 줌
#
# 수업에서 배운 스타일과 다른 점
#   - 손실을 모델 바깥(criterion)이 아니라 모델 안의 compute_loss()에서 계산
#     → 학습 루프는 정책 종류를 몰라도 model.compute_loss()만 부르면 됨
#   - forward() 대신 sample_actions()가 "추론" 역할
# ============================================================================

from __future__ import annotations

import abc  # Abstract Base Class : "자식 클래스가 꼭 구현해야 하는 함수"를 정하는 도구
from typing import Literal, TypeAlias  # 타입 표시용 (실행에는 영향 없음)

import torch
from torch import nn


# ----------------------------------------------------------------------------
# BasePolicy : 부모 클래스 (규칙서)
#   nn.Module을 상속 → PyTorch 모델의 기능(parameters(), to(device), train()/eval())을 그대로 가짐
#   @abc.abstractmethod가 붙은 두 함수를 자식이 구현하지 않으면 객체 생성 자체가 에러남
# ----------------------------------------------------------------------------
class BasePolicy(nn.Module, metaclass=abc.ABCMeta):
    """Base class for action chunking policies."""

    def __init__(self, state_dim: int, action_dim: int, chunk_size: int) -> None:
        super().__init__()  # nn.Module 초기화 (이게 있어야 층을 self.xxx에 등록 가능)
        self.state_dim = state_dim  # 상태 개수 (Push-T: 5)
        self.action_dim = action_dim  # 행동 개수 (Push-T: 2)
        self.chunk_size = chunk_size  # 한 번에 예측할 행동 스텝 수 K (기본 8)

    @abc.abstractmethod
    def compute_loss(
        self, state: torch.Tensor, action_chunk: torch.Tensor
    ) -> torch.Tensor:
        """Compute training loss for a batch."""
        # 학습용 : (상태, 정답 행동 묶음) → 손실 스칼라 하나. train.py가 호출

    @abc.abstractmethod
    def sample_actions(
        self,
        state: torch.Tensor,
        *,  # 이 뒤의 인자는 반드시 이름=값 형태로 넘겨야 함
        num_steps: int = 10,  # only applicable for flow policy
    ) -> torch.Tensor:
        """Generate a chunk of actions with shape (batch, chunk_size, action_dim)."""
        # 추론용 : 상태 → 행동 묶음 (B, K, 2). evaluation.py가 호출


# ----------------------------------------------------------------------------
# MSEPolicy : HW1 1부
#   입력 : state (B, 5)
#   처리 : MLP 5 → 256 → 256 → 256 → 16, 결과를 (B, 8, 2)로 모양 변경
#   출력 : 예측 행동 묶음 (B, 8, 2)
#   손실 : (예측 - 정답)² 의 평균 = PDF 식 (1)
#   한계 : MSE는 정답들의 "평균"을 고르므로, 정답이 여러 개(다봉)면 어중간한 행동이 나옴
# ----------------------------------------------------------------------------
class MSEPolicy(BasePolicy):
    """Predicts action chunks with an MSE loss."""

    ### TODO: IMPLEMENT MSEPolicy HERE ###
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        chunk_size: int,
        hidden_dims: tuple[int, ...] = (128, 128),  # 기본값이지만 train.py가 (256, 256, 256)을 넘겨 줌
    ) -> None:
        super().__init__(state_dim, action_dim, chunk_size)  # 부모가 self.state_dim 등을 저장
        # 은닉층 크기 목록을 따라 층을 차례로 쌓는다
        #
        # [복습] nn.Linear(in, out) : 완전연결층. 출력 = 입력 @ W^T + b
        #   - W 모양 (out, in), b 모양 (out,) → 학습되는 파라미터 수 = in*out + out
        #   - 예) nn.Linear(5, 256) : 5*256 + 256 = 1,536개
        #   - 입력 (B, in) → 출력 (B, out). 배치 B개를 한 번에 계산
        # [복습] nn.ReLU() : 활성화 함수. 출력 = max(0, x) (음수는 0, 양수는 그대로)
        #   - 왜 필요? 선형층만 여러 개 쌓으면 결국 선형층 하나와 같아짐.
        #     중간에 비선형 함수를 끼워야 곡선 같은 복잡한 관계를 배울 수 있음
        #   - 파라미터 없음 (학습되는 값이 없는 층)
        layers = []
        in_dim = state_dim  # 첫 층의 입력 개수 = 상태 개수 (5)
        for h in hidden_dims:  # 예) (256, 256, 256)이면 3번 반복
            layers += [nn.Linear(in_dim, h), nn.ReLU()]  # 선형층 + 활성화 함수를 한 쌍씩 추가
            in_dim = h  # 다음 층의 입력 = 이번 층의 출력
        layers.append(nn.Linear(in_dim, chunk_size * action_dim))  # 마지막 층: 8×2=16개를 한 줄로 출력 (활성화 없음 → 음수도 가능)
        self.net = nn.Sequential(*layers)  # 리스트를 풀어서(*) 하나의 신경망으로 묶기. self.net(x)로 순전파
        # [복습] nn.Sequential : 층들을 순서대로 이어 붙인 모델. self.net(x)는
        #   x → Linear → ReLU → Linear → ReLU → Linear → ReLU → Linear 순서로 통과시킨 결과.
        #   forward()를 직접 쓰는 대신 이걸 쓰면 층 개수가 바뀌어도 코드를 안 고쳐도 됨.
        #   self.net에 저장하면 nn.Module이 안의 파라미터를 자동 등록 → model.parameters()에 포함됨
        #   (기본 설정 기준 전체 파라미터: 1,536 + 65,792 + 65,792 + 4,112 = 137,232개)

    def compute_loss(
        self,
        state: torch.Tensor,  # (B, 5) 정규화된 상태
        action_chunk: torch.Tensor,  # (B, 8, 2) 정규화된 전문가 행동 묶음 (정답)
    ) -> torch.Tensor:
        pred = self.sample_actions(state)  # 예측 (B, 8, 2)
        return ((pred - action_chunk) ** 2).mean()  # 원소별 오차 제곱의 평균 = nn.MSELoss()와 같은 값
        # [복습] 텐서 연산은 원소별(element-wise)로 일어남
        #   pred - action_chunk : 같은 위치끼리 뺄셈, 모양 (B, 8, 2) 그대로
        #   ** 2                : 원소마다 제곱
        #   .mean()             : 전체 B*8*2개 숫자의 평균 → 숫자 하나(스칼라 텐서)
        #   손실이 스칼라여야 loss.backward()로 역전파할 수 있음

    def sample_actions(
        self,
        state: torch.Tensor,  # (B, 5)
        *,
        num_steps: int = 10,  # MSE 정책에서는 안 씀 (flow 정책과 호출 방법을 맞추려고 있는 자리)
    ) -> torch.Tensor:
        out = self.net(state)  # (B, 5) → (B, 16)
        return out.view(-1, self.chunk_size, self.action_dim)  # (B, 16) → (B, 8, 2)
        # -1은 "이 자리는 알아서 계산해라". 학습 때 B=128, 평가 때 B=1이라 숫자로 고정하면 안 됨
        # [복습] .view(a, b, c) : 숫자 개수는 그대로 두고 모양만 바꿈 (numpy의 reshape와 같음)
        #   (128, 16) = 2048개 → (128, 8, 2) = 2048개. 한 줄의 16개를 [행동1 x,y], [행동2 x,y]... 8쌍으로 나눔
        #   메모리가 연속이 아니면 에러가 날 수 있는데, 그럴 땐 .reshape()를 쓰면 됨


# ----------------------------------------------------------------------------
# FlowMatchingPolicy : HW1 2부 (아직 구현 전)
#   입력 : state (B, 5)
#   학습 : 정답 A와 노이즈 A0 사이의 중간 지점 A_τ = A0 + τ(A - A0) 에서
#          신경망이 정답 속도 (A - A0)를 맞히도록 MSE 학습   (PDF 식 2)
#   추론 : 노이즈에서 출발해 "위치 += (1/n) × 예측 속도"를 n번 반복   (PDF 식 3)
#   출력 : 행동 묶음 (B, 8, 2)
#   신경망 입력 : 상태(5) + 지금 위치(8×2=16) + τ(1) = 22개를 이어 붙여서 넣으면 됨
# ----------------------------------------------------------------------------
class FlowMatchingPolicy(BasePolicy):
    """Predicts action chunks with a flow matching loss."""

    ### TODO: IMPLEMENT FlowMatchingPolicy HERE ###
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        chunk_size: int,
        hidden_dims: tuple[int, ...] = (128, 128),
    ) -> None:
        super().__init__(state_dim, action_dim, chunk_size)
        layers = []
        in_dim = state_dim + chunk_size * action_dim + 1  # 빈칸 1: 입력 개수 (상태 + 지금 위치 + τ)
        for h in hidden_dims:
            layers += [nn.Linear(in_dim, h), nn.ReLU()]
            in_dim = h
        layers.append(nn.Linear(in_dim, chunk_size * action_dim))  # 출력: 속도 16개
        self.net = nn.Sequential(*layers)

    def _velocity(self, state, a_tau, tau):
        # 신경망에 (상태, 지금 위치, τ)를 넣어 속도를 예측
        x = torch.cat([state, a_tau.flatten(1), tau.view(-1, 1)], dim=1)   # (B, 22)
        return self.net(x).view(-1, self.chunk_size, self.action_dim)      # (B, 8, 2)

    def compute_loss(
        self,
        state: torch.Tensor,
        action_chunk: torch.Tensor,
    ) -> torch.Tensor:
        B = state.shape[0]
        a0 = torch.randn_like(action_chunk)                     # 출발점: 노이즈 (B, 8, 2)
        tau = torch.rand(B, 1, 1, device=state.device)          # 진행도: 0~1
        a_tau = a0 + tau * (action_chunk - a0)                  # 빈칸 2: 중간 지점
        target = action_chunk - a0                              # 빈칸 3: 정답 속도
        v_pred = self._velocity(state, a_tau, tau)
        return ((v_pred - target) ** 2).mean()                  # PDF 식 (2)

    def sample_actions(
        self,
        state: torch.Tensor,
        *,
        num_steps: int = 10,
    ) -> torch.Tensor:
        B = state.shape[0]
        a = torch.randn(B, self.chunk_size, self.action_dim, device=state.device)  # 출발: 노이즈
        dt = 1.0 / num_steps                                    # 한 걸음 크기
        for i in range(num_steps):
            tau = torch.full((B, 1, 1), i * dt, device=state.device)   # 빈칸 4: 지금 τ 값
            a = a + dt * self._velocity(state, a, tau)                 # 빈칸 5: 한 걸음 이동 (PDF 식 3)
        return a


# 정책 종류 이름은 "mse" 또는 "flow"만 허용한다는 타입 표시 (--policy-type 옵션의 선택지가 됨)
PolicyType: TypeAlias = Literal["mse", "flow"]


# ----------------------------------------------------------------------------
# build_policy : 정책 공장
#   입력 : 정책 이름 + 크기 정보
#   출력 : MSEPolicy 또는 FlowMatchingPolicy 객체
#   train.py는 이 함수만 부르면 되므로, 정책 종류가 늘어도 train.py는 안 고쳐도 됨
# ----------------------------------------------------------------------------
def build_policy(
    policy_type: PolicyType,
    *,
    state_dim: int,
    action_dim: int,
    chunk_size: int,
    hidden_dims: tuple[int, ...] = (128, 128),
) -> BasePolicy:
    if policy_type == "mse":
        return MSEPolicy(
            state_dim=state_dim,
            action_dim=action_dim,
            chunk_size=chunk_size,
            hidden_dims=hidden_dims,
        )
    if policy_type == "flow":
        return FlowMatchingPolicy(
            state_dim=state_dim,
            action_dim=action_dim,
            chunk_size=chunk_size,
            hidden_dims=hidden_dims,
        )
    raise ValueError(f"Unknown policy type: {policy_type}")  # 둘 다 아니면 에러
