"""Evaluation utilities for Push-T policies."""

# ============================================================================
# evaluation.py : 평가와 기록 담당 파일 (과제에서는 고칠 필요 없음, 호출만 함)
# ----------------------------------------------------------------------------
# 역할 : 학습 중인 정책을 Push-T 시뮬레이터에서 실제로 100판 돌려 점수를 매기고,
#        점수·영상·모델 체크포인트를 log.csv와 W&B에 기록한다.
# 입력 : 정책(model), 정규화 도구, 현재 학습 스텝 등
# 처리 : 판마다 [관측 → 정규화 → 정책이 8개 행동 예측 → 역정규화 → 8스텝 실행]을
#        끝날 때까지 반복 (= 액션 청킹, 묶음 안에서는 관측을 다시 보지 않음)
# 출력 : eval/mean_reward (100판의 최고 점수 평균), 영상 5개, 체크포인트 .pkl
#
# 지도학습과 다른 점 : CIFAR-10의 test()는 고정된 테스트 이미지만 채점했지만,
#   여기서는 정책의 행동이 다음 관측을 바꾼다. 1·2장의 "분포 이동"이 실제로 일어나는 곳.
# ============================================================================

from __future__ import annotations

import io  # 메모리 속 바이트를 파일처럼 다루기 (영상 업로드용)
import os
import tempfile  # 임시 파일 만들기
from pathlib import Path

import gym_pusht  # noqa: F401   # import만 해도 Push-T 환경이 gymnasium에 등록됨 (직접 쓰지는 않음)
import gymnasium as gym  # 강화학습 환경 표준 인터페이스 (reset, step)
import imageio.v2 as imageio  # 이미지 프레임 → mp4 영상
import numpy as np
import torch
import wandb
import shutil  # 폴더 통째로 복사
from PIL import Image  # 이미지 크기 조절

from hw1_imitation.data import Normalizer
from hw1_imitation.model import BasePolicy
import copy
from typing import Any

ENV_ID = "gym_pusht/PushT-v0"  # 사용할 환경 이름
NUM_EVAL_EPISODES = 100  # 평가 한 번에 돌리는 판 수


# ----------------------------------------------------------------------------
# Logger : 기록 도구
#   log()            : 숫자는 log.csv에, 숫자+영상은 W&B에 동시에 기록
#   dump_for_grading(): 학습이 끝나면 W&B 로컬 폴더를 exp/<실험>/wandb 로 복사
# ----------------------------------------------------------------------------
class Logger:
    """Logger for logging metrics."""

    # CSV에는 숫자만 쓸 수 있으므로, 이미지/영상/히스토그램은 CSV에서 제외
    CSV_DISALLOWED_TYPES = (wandb.Image, wandb.Video, wandb.Histogram)

    def __init__(self, path: Path):
        if path.exists():  # 같은 이름 폴더가 있으면 덮어쓰지 않고 에러 (기록 보호)
            raise FileExistsError(f"Log directory {path} already exists.")
        path.mkdir(parents=True)
        self.path = path
        self.csv_path = path / "log.csv"
        self.header = None  # CSV 첫 줄(열 이름). 첫 기록 때 정해짐
        self.rows = []  # 지금까지 기록한 내용 사본

    def log(self, row: dict[str, Any], step: int) -> None:
        # 입력 예) row = {"train/loss": 0.167}, step = 100
        row["step"] = step
        if self.header is None:  # 첫 기록이면
            # 숫자 항목의 이름들로 CSV 헤더를 만든다
            self.header = [
                k
                for k, v in row.items()
                if not isinstance(v, self.CSV_DISALLOWED_TYPES)
            ]
            with self.csv_path.open("w") as f:  # "w": 새로 쓰기
                f.write(",".join(self.header) + "\n")  # 예) "train/loss,step"
        # 영상 등을 뺀 숫자 항목만 남김
        filtered_row = {
            k: v for k, v in row.items() if not isinstance(v, self.CSV_DISALLOWED_TYPES)
        }
        with self.csv_path.open("a") as f:  # "a": 이어 쓰기
            f.write(
                # 헤더 순서대로 값을 쓰고, 없는 항목은 빈칸
                # 주의: 헤더는 첫 기록(train/loss)으로 정해져서, 나중에 오는 eval/mean_reward는
                #       CSV에 열이 없을 수 있음 → 그래프를 그릴 땐 W&B 기록도 같이 확인
                ",".join([str(filtered_row.get(k, "")) for k in self.header]) + "\n"
            )
        wandb.log(row, step=step)  # W&B에는 영상까지 전부 기록
        self.rows.append(copy.deepcopy(row))

    def dump_for_grading(self) -> None:
        wandb_dir = Path(wandb.run.dir).parent  # 이번 W&B 실행의 로컬 폴더 (wandb/run-...)
        wandb.finish()  # W&B 실행 종료 (남은 데이터 업로드)
        shutil.copytree(wandb_dir, self.path / "wandb")  # exp/<실험>/wandb 로 복사


# 프레임(이미지 배열) 크기를 바꿈. 입력 (H, W, 3) → 출력 (size[1], size[0], 3)
def resize_frame(frame: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    image = Image.fromarray(frame)  # numpy → PIL 이미지
    resized = image.resize(size, resample=Image.BILINEAR)  # 쌍선형 보간으로 크기 조절
    return np.asarray(resized)  # 다시 numpy로


# ----------------------------------------------------------------------------
# encode_video : 프레임 목록 → W&B 영상 객체
#   처리 : 임시 mp4 파일에 쓰기 → 바이트로 읽기 → 임시 파일 삭제
# ----------------------------------------------------------------------------
def encode_video(frames: list[np.ndarray], fps: int = 20) -> wandb.Video | None:
    if not frames:  # 프레임이 없으면 영상도 없음
        return None

    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
        tmp_path = tmp.name  # 임시 파일 경로만 받아 둠

    try:
        with imageio.get_writer(
            tmp_path,
            fps=fps,  # 초당 20프레임
            codec="libx264",  # 일반적인 mp4 압축 방식
            macro_block_size=1,  # 해상도가 16의 배수가 아니어도 되게
        ) as writer:
            for frame in frames:
                writer.append_data(frame)  # 프레임 한 장씩 추가
        with open(tmp_path, "rb") as f:  # "rb": 바이너리로 읽기
            video_bytes = f.read()
        return wandb.Video(io.BytesIO(video_bytes), format="mp4")
    finally:  # 성공하든 에러가 나든 임시 파일은 지움
        try:
            os.remove(tmp_path)
        except FileNotFoundError:
            pass


# ----------------------------------------------------------------------------
# log_checkpoint_artifact : 모델 저장
#   처리 : 모델 전체를 .pkl로 저장하고 W&B "아티팩트"(버전 관리되는 파일)로 업로드
# ----------------------------------------------------------------------------
def log_checkpoint_artifact(model: BasePolicy, step: int) -> None:
    if wandb.run is None:  # wandb.init()을 안 했으면 저장할 곳이 없음
        raise RuntimeError("wandb.init did not create a run.")

    run_dir = Path(wandb.run.dir)
    checkpoint_dir = run_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / f"checkpoint_step_{step}.pkl"
    torch.save(model, checkpoint_path)  # 가중치만이 아니라 모델 객체 통째로 저장

    artifact = wandb.Artifact(
        name=f"policy-checkpoint-{wandb.run.id}",
        type="model",
        metadata={"step": step},  # 몇 스텝째 모델인지 메모
    )
    artifact.add_file(checkpoint_path.as_posix(), name=checkpoint_path.name)
    wandb.log_artifact(artifact)


# ----------------------------------------------------------------------------
# evaluate_policy : 평가 (train.py가 1만 스텝마다 + 마지막에 호출)
#   입력 : 정책, 정규화 도구, 장치, K, 영상 설정, flow 걸음 수, 현재 스텝, logger
#   출력 : 없음 (대신 eval/mean_reward, 영상, 체크포인트가 기록됨)
# ----------------------------------------------------------------------------
def evaluate_policy(
    model: BasePolicy,
    normalizer: Normalizer,
    device: torch.device,
    chunk_size: int,
    video_size: tuple[int, int],
    num_video_episodes: int,
    flow_num_steps: int,
    step: int,
    logger: Logger,
) -> None:
    """Evaluate a policy in the Push-T environment and log results to Weights & Biases.

    This function runs a fixed number of evaluation episodes in the Push-T gym
    environment using the provided policy. It normalizes observations with the
    given normalizer, requests a chunk of actions from the policy (optionally
    using multiple sampling steps for flow-based policies), and executes those
    actions in the environment until each episode terminates.

    Metrics:
        - Logs the mean of per-episode maximum reward.
        - Optionally logs rendered rollout videos for the first
          ``num_video_episodes`` episodes.

    Checkpointing:
        - Saves the policy as a ``.pkl`` file and uploads it as a W&B artifact
          tagged with the current training step.

    Args:
        model: The policy to evaluate.
        normalizer: Normalizer used to scale states and actions.
        device: Device on which to run policy inference.
        chunk_size: Number of actions to generate per policy call.
        video_size: (width, height) for rendered rollout videos.
        num_video_episodes: How many episodes to record videos for.
        flow_num_steps: Number of denoising steps used by flow policies.
        step: Training step used for logging and artifact metadata.
        logger: Logger for logging metrics.
    """
    model.eval()  # 평가 모드 (드롭아웃/배치정규화가 있다면 추론용으로 전환)
    rewards: list[float] = []  # 판마다의 점수
    videos: list[wandb.Video] = []

    # Push-T 환경 만들기. obs_type="state": 이미지 대신 숫자 5개를 관측으로 받음
    env = gym.make(ENV_ID, obs_type="state", render_mode="rgb_array")
    action_low = env.action_space.low  # 행동(목표 좌표)의 최솟값
    action_high = env.action_space.high  # 행동의 최댓값

    for ep_idx in range(NUM_EVAL_EPISODES):  # 100판
        obs, _ = env.reset(seed=ep_idx)  # 판마다 다른 시드 → T의 시작 위치가 달라짐 (매번 같은 100판)
        done = False
        chunk_index = chunk_size  # "묶음을 다 썼음" 상태로 시작 → 첫 스텝에 바로 예측하게
        action_chunk: np.ndarray | None = None
        frames: list[np.ndarray] = []
        max_reward = 0.0  # 이번 판에서 받은 가장 높은 점수
        save_video = ep_idx < num_video_episodes  # 앞의 5판만 영상 저장

        while not done:  # 판이 끝날 때까지 한 스텝씩
            # 묶음을 다 실행했으면 새 묶음 예측 (8스텝에 한 번)
            if action_chunk is None or chunk_index >= chunk_size:
                state = (
                    torch.from_numpy(normalizer.normalize_state(obs)).float().to(device)  # (5,)
                )
                # [복습] torch.no_grad() : 이 블록 안에서는 계산 그래프를 기록하지 않음
                #   학습 때는 backward()를 위해 모든 연산을 기록하지만, 평가 때는 필요 없음
                #   model.eval()과는 다른 것! eval()은 층의 동작 모드, no_grad()는 기울기 기록 여부
                with torch.no_grad():  # 추론만 하므로 기울기 계산 끔 (빠르고 메모리 절약)
                    pred_chunk = (
                        model.sample_actions(
                            state.unsqueeze(0), num_steps=flow_num_steps  # (5,) → (1, 5) 배치 차원 추가
                        )
                        .cpu()  # GPU → CPU (numpy는 GPU 메모리를 직접 못 읽음)
                        .numpy()[0]  # 텐서 → numpy, 배치 차원 제거 → (8, 2)
                        # [복습] unsqueeze(0) ↔ [0] : 모델은 항상 배치 (B, ...)를 기대하므로
                        #   한 개짜리 입력은 앞에 차원을 하나 붙여 B=1로 만들고, 결과에서 다시 떼어 냄
                    )
                action_chunk = normalizer.denormalize_action(pred_chunk)  # 정규화 해제 → 실제 좌표
                action_chunk = np.clip(action_chunk, action_low, action_high)  # 범위 밖이면 잘라냄
                chunk_index = 0  # 묶음의 첫 행동부터

            action = action_chunk[chunk_index]  # 이번 스텝에 실행할 행동 (2,)
            obs, reward, terminated, truncated, info = env.step(  # 환경에 행동 입력 → 다음 관측, 보상
                action.astype(np.float32)
            )
            # reward : T가 목표 영역과 겹친 비율 (0~1)
            # terminated : 성공해서 끝남, truncated : 시간 제한으로 끝남
            if save_video:
                frame = env.render()  # 현재 화면을 이미지로
                frame = resize_frame(frame, video_size)
                frames.append(frame)
            max_reward = max(max_reward, float(reward))  # 이번 판 최고 점수 갱신
            done = terminated or truncated
            chunk_index += 1  # 묶음의 다음 행동으로

        rewards.append(max_reward)  # 판 점수 = 그 판에서 가장 잘 맞췄던 순간의 점수
        if save_video:
            video = encode_video(frames, fps=20)
            if video is not None:
                videos.append(video)

    env.close()
    # 100판 점수의 평균 → 과제 목표: MSE ≥ 0.5, flow ≥ 0.7
    log_data: dict[str, float | wandb.Video] = {
        "eval/mean_reward": float(np.mean(rewards))
    }
    for idx, video in enumerate(videos):
        log_data[f"eval/rollout_ep{idx}"] = video  # 영상 5개
    logger.log(log_data, step=step)
    log_checkpoint_artifact(model, step=step)  # 이 시점의 모델 저장
