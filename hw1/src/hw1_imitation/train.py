"""Train and evaluate a Push-T imitation policy."""

# ============================================================================
# train.py : 학습 실행 파일 (여기가 시작점)
# ----------------------------------------------------------------------------
# 역할 : 설정 읽기 → 데이터 준비 → 정책 만들기 → 학습 루프 → 주기적 평가 → 기록 저장
# 입력 : 터미널 옵션 (예: --policy-type mse --num-epochs 400 --exp-name mse)
# 처리 : 일반 PyTorch 학습 루프 + 일정 스텝마다 Push-T를 실제로 실행해 점수 측정
# 출력 : exp/<실험이름>/log.csv, exp/<실험이름>/wandb/ (그래프, 영상, 체크포인트)
#
# 실행 예 : uv run src/hw1_imitation/train.py --policy-type mse --exp-name mse
# 옵션 확인 : uv run src/hw1_imitation/train.py --help
# ============================================================================

from __future__ import annotations

from dataclasses import asdict, dataclass  # asdict: 데이터클래스 → 딕셔너리
from datetime import datetime  # 실험 이름에 날짜/시간 붙이기
from pathlib import Path
from typing import Any

import numpy as np
import torch
import tyro  # 데이터클래스를 보고 터미널 옵션(--lr 등)을 자동으로 만들어 주는 라이브러리
import wandb  # 실험 기록 (그래프, 영상, 설정)
from torch.utils.data import DataLoader

# 같은 폴더의 다른 파일에서 필요한 것들을 가져옴
from hw1_imitation.data import (
    Normalizer,
    PushtChunkDataset,
    download_pusht,
    load_pusht_zarr,
)
from hw1_imitation.model import build_policy, PolicyType
from hw1_imitation.evaluation import Logger, evaluate_policy

LOGDIR_PREFIX = "exp"  # 실험 기록을 저장할 최상위 폴더 이름


# ----------------------------------------------------------------------------
# TrainConfig : 하이퍼파라미터 모음
#   여기 적힌 변수 하나하나가 터미널 옵션이 된다. (밑줄 _ 은 하이픈 - 으로 바뀜)
#   예) num_epochs → --num-epochs, hidden_dims → --hidden-dims 512 512
# ----------------------------------------------------------------------------
@dataclass
class TrainConfig:
    # The path to download the Push-T dataset to.
    data_dir: Path = Path("data")  # 데이터를 받을 폴더

    # The policy type -- either MSE or flow.
    policy_type: PolicyType = "mse"  # "mse" 또는 "flow"
    # The number of denoising steps to use for the flow policy (has no effect for the MSE policy).
    flow_num_steps: int = 10  # flow 정책의 오일러 걸음 수 n
    # The action chunk size.
    chunk_size: int = 8  # K, 한 번에 예측/실행할 행동 수

    # [복습] 배치 / 스텝 / 에폭
    #   배치 : 한 번에 모델에 넣는 샘플 묶음 (128개)
    #   스텝 : 배치 하나로 가중치를 한 번 업데이트하는 것
    #   에폭 : 전체 데이터를 한 바퀴 다 보는 것 (= 데이터 수 ÷ 배치 크기 만큼의 스텝)
    batch_size: int = 128  # 한 번 업데이트에 쓰는 샘플 수 (크면 기울기가 안정적, 작으면 업데이트가 잦음)
    lr: float = 3e-4  # 학습률 = 한 스텝에 가중치를 얼마나 크게 움직일지 (0.0003, Adam에서 흔히 쓰는 값)
    weight_decay: float = 0.0  # L2 정규화 세기. 가중치가 너무 커지지 않게 매 스텝 조금씩 줄임 (과적합 억제, 0이면 안 씀)
    hidden_dims: tuple[int, ...] = (256, 256, 256)  # MLP 은닉층 크기들 → 은닉층 3개, 각 256
    # The number of epochs to train for.
    num_epochs: int = 400  # 전체 데이터를 몇 바퀴 돌지 (1에폭 ≈ 189스텝 → 총 약 7.5만 스텝)
    # How often to run evaluation, measured in training steps.
    eval_interval: int = 10_000  # 몇 스텝마다 Push-T로 평가할지 (10_000은 10000과 같음)
    num_video_episodes: int = 5  # 평가 때 영상으로 남길 판 수
    video_size: tuple[int, int] = (256, 256)  # 영상 해상도
    # How often to log training metrics, measured in training steps.
    log_interval: int = 100  # 몇 스텝마다 손실을 기록할지
    # Random seed.
    seed: int = 42  # 난수 시드 (같으면 결과가 거의 재현됨). 가중치 초기값, 셔플 순서 등이 난수로 정해지기 때문
    # WandB project name.
    wandb_project: str = "hw1-imitation"  # W&B 웹에서 보이는 프로젝트 이름
    # Experiment name suffix for logging and WandB.
    exp_name: str | None = None  # 실험 이름 꼬리표 (예: "mse" → seed_42_날짜_mse)


# ----------------------------------------------------------------------------
# parse_train_config : 터미널 옵션 읽기
#   입력 : 터미널 인자 (없으면 실제 터미널에서 읽음)
#   출력 : 값이 채워진 TrainConfig
# ----------------------------------------------------------------------------
def parse_train_config(
    args: list[str] | None = None,
    *,
    defaults: TrainConfig | None = None,  # 기본값을 바꾸고 싶을 때 (modal_train.py가 사용)
    description: str = "Train a Push-T MLP policy.",  # --help에 보이는 설명
) -> TrainConfig:
    defaults = defaults or TrainConfig()  # 안 주면 위의 기본값 사용
    return tyro.cli(  # 옵션을 해석해서 TrainConfig를 만들어 줌
        TrainConfig,
        args=args,
        default=defaults,
        description=description,
    )


# 시드 고정 : numpy, torch(CPU), torch(GPU)의 난수를 모두 같은 값에서 시작시킴
def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# W&B에 설정을 저장하려면 딕셔너리여야 하고, Path 객체는 문자열로 바꿔야 함
def config_to_dict(config: TrainConfig) -> dict[str, Any]:
    data = asdict(config)
    for key, value in data.items():
        if isinstance(value, Path):
            data[key] = str(value)
    return data


# ----------------------------------------------------------------------------
# run_training : 학습 전체
#   입력 : TrainConfig
#   출력 : 없음 (대신 exp/ 와 W&B에 기록이 남음)
# ----------------------------------------------------------------------------
def run_training(config: TrainConfig) -> None:
    set_seed(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # GPU 있으면 GPU
    # [복습] device 규칙 : 모델과 데이터는 반드시 같은 장치에 있어야 계산 가능.
    #   다르면 "Expected all tensors to be on the same device" 에러. 그래서 아래에서
    #   model.to(device), state.to(device)를 둘 다 한다
    print(f"Using device: {device}")

    # ---- 1. 데이터 준비 ----
    zarr_path = download_pusht(config.data_dir)  # 없으면 다운로드
    states, actions, episode_ends = load_pusht_zarr(zarr_path)  # (N,5), (N,2), 에피소드 끝 위치
    normalizer = Normalizer.from_data(states, actions)  # 전체 데이터로 평균/표준편차 계산

    dataset = PushtChunkDataset(  # dataset[i] → (state (5,), action_chunk (8, 2))
        states,
        actions,
        episode_ends,
        chunk_size=config.chunk_size,
        normalizer=normalizer,
    )

    # [복습] DataLoader : Dataset에서 샘플을 꺼내 배치로 쌓아 주는 반복자
    #   for 문에서 한 번 돌 때마다 dataset[i]를 128번 호출하고, 결과를 쌓아서
    #   (5,) 128개 → (128, 5),  (8, 2) 128개 → (128, 8, 2) 로 만들어 준다
    loader = DataLoader(  # 샘플을 배치로 묶어 줌 → (128, 5), (128, 8, 2)
        dataset,
        batch_size=config.batch_size,
        shuffle=True,  # 에폭마다 순서 섞기. 안 섞으면 같은 에피소드의 비슷한 샘플이 한 배치에 몰려 학습이 치우침
        drop_last=True,  # 마지막에 128개가 안 되는 자투리 배치는 버림 (배치 크기를 항상 같게)
    )

    # ---- 2. 정책 만들기 ----
    model = build_policy(
        config.policy_type,  # "mse" 또는 "flow"
        state_dim=states.shape[1],  # 5
        action_dim=actions.shape[1],  # 2
        chunk_size=config.chunk_size,  # 8
        hidden_dims=config.hidden_dims,  # (256, 256, 256)
    ).to(device)  # 모델 가중치를 GPU(또는 CPU)로 옮김

    # ---- 3. 기록 준비 ----
    # 실험 이름 예) seed_42_20261109_173437_smoke
    exp_name = f"seed_{config.seed}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    if config.exp_name is not None:
        exp_name += f"_{config.exp_name}"
    log_dir = Path(LOGDIR_PREFIX) / exp_name  # exp/seed_42_..._smoke
    wandb.init(  # W&B에 새 실험(run)을 만들고 설정을 함께 저장
        project=config.wandb_project, config=config_to_dict(config), name=exp_name
    )
    logger = Logger(log_dir)  # log.csv + W&B에 동시에 기록하는 도구 (evaluation.py)

    # ---- 4. 학습 루프 ----
    ### TODO: PUT YOUR MAIN TRAINING LOOP HERE ###
    # [복습] 옵티마이저 : 계산된 기울기를 보고 가중치를 실제로 바꾸는 담당
    #   SGD  : 가중치 ← 가중치 - lr × 기울기   (가장 기본)
    #   Adam : SGD에 두 가지를 더함
    #     ① 모멘텀 : 최근 기울기들의 평균 방향으로 이동 → 지그재그가 줄고 꾸준히 내려감
    #     ② 적응형 학습률 : 기울기가 자주 큰 파라미터는 보폭을 줄이고, 작은 파라미터는 키움
    #   그래서 lr을 대충 정해도 잘 동작하는 편이라 기본 선택으로 많이 씀
    #   model.parameters() : 모델 안의 모든 학습 가능한 가중치(W, b)를 옵티마이저에 넘김
    optimizer = torch.optim.Adam(  # 수업에서 쓰던 것과 같은 Adam
        model.parameters(), lr=config.lr, weight_decay=config.weight_decay
    )
    step = 0  # 지금까지 업데이트한 횟수 (에폭과 무관하게 계속 증가)
    for epoch in range(config.num_epochs):  # 전체 데이터를 num_epochs 바퀴
        for state, action_chunk in loader:  # 배치 하나씩: (128, 5), (128, 8, 2)
            state = state.to(device)  # 데이터도 모델과 같은 장치로
            action_chunk = action_chunk.to(device)

            # [복습] 학습 한 스텝 = 아래 4줄 (순전파 → 기울기 초기화 → 역전파 → 업데이트)
            loss = model.compute_loss(state, action_chunk)  # ① 순전파 + 손실 (MSE면 식 1). 계산 과정이 그래프로 기록됨
            optimizer.zero_grad()  # ② 이전 스텝의 기울기 지우기. PyTorch는 기울기를 "더해서" 쌓기 때문에 안 지우면 섞임
            loss.backward()  # ③ 역전파: 연쇄 법칙으로 "손실을 각 가중치로 미분한 값"을 계산해 각 파라미터의 .grad에 저장
            optimizer.step()  # ④ .grad를 보고 Adam 규칙대로 가중치 업데이트 → 손실이 줄어드는 방향으로 한 걸음
            step+=1

            if step % config.log_interval == 0:  # 100스텝마다 손실 기록
                logger.log({"train/loss": loss.item()}, step=step)  # .item(): 텐서 → 파이썬 숫자
            if step % config.eval_interval == 0:  # 1만 스텝마다 실제 Push-T로 평가
                evaluate_policy(
                    model=model, normalizer=normalizer, device=device,
                    chunk_size=config.chunk_size, video_size=config.video_size,
                    num_video_episodes=config.num_video_episodes,
                    flow_num_steps=config.flow_num_steps,
                    step=step, logger=logger,
                )
                model.train()  # 평가 함수가 model.eval()로 바꿔 놨으니 학습 모드로 되돌림
                # [복습] train()/eval() 모드 : Dropout(학습 때만 뉴런을 무작위로 끔)과
                #   BatchNorm(학습 때는 배치 통계, 평가 때는 저장된 통계 사용)의 동작을 바꿈.
                #   지금 MLP에는 둘 다 없어서 결과는 같지만, 습관적으로 되돌리는 게 안전함

    # 학습이 끝난 뒤 마지막 평가 한 번 (최종 점수)
    evaluate_policy(
        model=model, normalizer=normalizer, device=device,
        chunk_size=config.chunk_size, video_size=config.video_size,
        num_video_episodes=config.num_video_episodes,
        flow_num_steps=config.flow_num_steps,
        step=step, logger=logger,
    )

    # W&B 실행을 끝내고, W&B 로컬 기록을 exp/<실험이름>/wandb 로 복사 (제출용 폴더 완성)
    logger.dump_for_grading()


def main() -> None:
    config = parse_train_config()  # 터미널 옵션 읽기
    run_training(config)  # 학습 시작


# 이 파일을 직접 실행했을 때만 main() 실행 (다른 파일이 import할 때는 실행 안 됨)
if __name__ == "__main__":
    main()