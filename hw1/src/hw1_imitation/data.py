"""Dataset utilities for Push-T."""

# ============================================================================
# data.py : 데이터 담당 파일
# ----------------------------------------------------------------------------
# 역할 : Push-T 전문가 시연 데이터를 내려받고, 읽고, 정규화해서
#        "(상태, 행동 묶음)" 쌍으로 잘라 DataLoader에 넘겨준다.
# 입력 : 인터넷의 pusht.zip (처음 한 번만 다운로드)
# 처리 : 다운로드 → 압축 해제 → zarr 파일 읽기 → 평균/표준편차 정규화
#        → 에피소드 경계를 넘지 않게 길이 K(chunk_size)짜리 행동 묶음으로 자르기
# 출력 : 학습용 샘플 하나 = (state (5,), action_chunk (K, 2)) 텐서 쌍
#
# 용어
#   - 에피소드 : 시연 한 판 (T를 목표 위치까지 민 기록 하나)
#   - state(5) : [로봇 x, 로봇 y, T블록 x, T블록 y, T블록 각도]
#   - action(2): 로봇이 다음에 갈 목표 위치 [x, y]
# ============================================================================

from __future__ import annotations  # 타입 표시를 문자열처럼 늦게 해석 (최신 문법 호환용)

import urllib.request  # URL에서 파일 다운로드
import zipfile  # zip 압축 해제
from dataclasses import dataclass  # 값만 담는 간단한 클래스를 쉽게 만드는 도구
from pathlib import Path  # 파일 경로를 객체로 다루기 ("data" / "pusht" 처럼 / 로 이어 붙임)

import numpy as np
import torch
import zarr  # 큰 배열을 디스크에 나눠 저장하는 포맷. Push-T 데이터가 이 형식
from torch.utils.data import Dataset  # PyTorch 데이터셋 부모 클래스

# 데이터 원본 주소 (Diffusion Policy 논문 저자들이 공개한 Push-T 시연)
PUSHT_URL = "https://diffusion-policy.cs.columbia.edu/data/training/pusht.zip"
# 압축을 풀면 생기는 zarr 폴더의 상대 경로
ZARR_RELATIVE_PATH = Path("pusht") / "pusht_cchi_v7_replay.zarr"


# ----------------------------------------------------------------------------
# Normalizer : 정규화 담당
#   입력 : 전체 states (N, 5), actions (N, 2)
#   처리 : 열(특징)마다 평균과 표준편차를 구해 저장
#   출력 : (x - 평균) / 표준편차 로 바꿔 주는 함수들
#   왜?  : 좌표는 0~500, 각도는 0~6처럼 단위가 제각각이라, 그대로 넣으면
#          큰 숫자 쪽만 학습된다. 모두 "평균 0, 표준편차 1" 근처로 맞춘다.
# ----------------------------------------------------------------------------
@dataclass(frozen=True)  # frozen=True : 한 번 만들면 값을 못 바꿈 (실수로 덮어쓰기 방지)
class Normalizer:
    """Feature-wise normalizer for states and actions."""

    state_mean: np.ndarray  # 상태 5개 각각의 평균, 모양 (5,)
    state_std: np.ndarray  # 상태 5개 각각의 표준편차, 모양 (5,)
    action_mean: np.ndarray  # 행동 2개 각각의 평균, 모양 (2,)
    action_std: np.ndarray  # 행동 2개 각각의 표준편차, 모양 (2,)

    @staticmethod  # self가 필요 없는 보조 함수
    def _safe_std(std: np.ndarray, eps: float = 1e-6) -> np.ndarray:
        # 표준편차가 0이면 나눗셈에서 무한대가 나오므로, 최소 1e-6으로 바닥을 깐다
        return np.maximum(std, eps)

    @classmethod  # 클래스 자체(cls)를 받아 새 Normalizer를 만들어 주는 "생성 함수"
    def from_data(cls, states: np.ndarray, actions: np.ndarray) -> "Normalizer":
        state_mean = states.mean(axis=0)  # axis=0 : 행(샘플) 방향으로 평균 → 특징별 평균 (5,)
        state_std = cls._safe_std(states.std(axis=0))  # 특징별 표준편차 (5,)
        action_mean = actions.mean(axis=0)  # (2,)
        action_std = cls._safe_std(actions.std(axis=0))  # (2,)
        return cls(state_mean, state_std, action_mean, action_std)  # Normalizer(...) 와 같음

    def normalize_state(self, state: np.ndarray) -> np.ndarray:
        # 표준화 : (값 - 평균) / 표준편차. 브로드캐스팅으로 (N,5)든 (5,)든 다 동작
        return (state - self.state_mean) / self.state_std

    def normalize_action(self, action: np.ndarray) -> np.ndarray:
        # 학습할 때 정답 행동도 같은 방식으로 정규화 → 모델은 "정규화된 행동"을 예측하게 됨
        return (action - self.action_mean) / self.action_std

    def denormalize_action(self, action: np.ndarray) -> np.ndarray:
        # 역변환 : 모델이 낸 정규화된 행동을 실제 좌표 단위로 되돌림 (평가 때 환경에 넣기 직전)
        return action * self.action_std + self.action_mean


# ----------------------------------------------------------------------------
# download_pusht : 데이터 다운로드
#   입력 : 저장할 폴더 (기본값 "data")
#   처리 : 이미 있으면 건너뛰고, 없으면 zip 다운로드 → 압축 해제
#   출력 : zarr 폴더 경로
# ----------------------------------------------------------------------------
def download_pusht(dataset_dir: Path) -> Path:
    """Download and extract the Push-T dataset if needed.

    Returns the path to the extracted Zarr dataset.
    """

    dataset_dir.mkdir(parents=True, exist_ok=True)  # 폴더 만들기 (이미 있어도 에러 안 냄)
    zarr_path = dataset_dir / ZARR_RELATIVE_PATH  # data/pusht/pusht_cchi_v7_replay.zarr
    if zarr_path.exists():  # 예전에 이미 받아 뒀으면
        return zarr_path  # 바로 반환 (두 번째 실행부터는 다운로드 안 함)

    zip_path = dataset_dir / "pusht.zip"
    if not zip_path.exists():  # zip도 없으면
        urllib.request.urlretrieve(PUSHT_URL, zip_path)  # 인터넷에서 내려받기

    with zipfile.ZipFile(zip_path, "r") as zip_ref:  # with : 끝나면 파일을 자동으로 닫음
        zip_ref.extractall(dataset_dir)  # data/ 아래에 압축 풀기

    return zarr_path


# ----------------------------------------------------------------------------
# load_pusht_zarr : 데이터 읽기
#   입력 : zarr 폴더 경로
#   출력 : states (N, 5), actions (N, 2), episode_ends (에피소드 개수,)
#   주의 : 모든 에피소드가 시간 순서대로 "한 줄로 이어 붙여져" 있다.
#          episode_ends가 각 에피소드가 끝나는 위치(다음 에피소드 시작 인덱스)를 알려 준다.
#          예) episode_ends = [161, 279, ...] → 0~160이 1판, 161~278이 2판 ...
# ----------------------------------------------------------------------------
def load_pusht_zarr(zarr_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    root = zarr.open(zarr_path, mode="r")  # 읽기 전용으로 열기 (딕셔너리처럼 접근)
    states = np.asarray(root["data"]["state"][:], dtype=np.float32)  # [:]로 전부 메모리에 올림
    actions = np.asarray(root["data"]["action"][:], dtype=np.float32)
    episode_ends = np.asarray(root["meta"]["episode_ends"][:], dtype=np.int64)  # 정수 인덱스
    return states, actions, episode_ends


# ----------------------------------------------------------------------------
# build_valid_indices : "묶음을 시작해도 되는 시점" 목록 만들기
#   입력 : episode_ends, chunk_size(K)
#   처리 : 시점 t에서 시작한 K개 행동 묶음이 같은 에피소드 안에 다 들어가는 t만 고름
#   출력 : 쓸 수 있는 시작 인덱스 배열
#   왜?  : 묶음이 에피소드 경계를 넘으면 "1판의 끝 + 2판의 처음"이 섞인
#          말도 안 되는 행동 묶음이 생긴다.
#   예)  : 1판이 0~160 (end=161), K=8 이면 마지막 시작점은 161-8=153
#          → 0, 1, ..., 153 (154개)이 유효
# ----------------------------------------------------------------------------
def build_valid_indices(episode_ends: np.ndarray, chunk_size: int) -> np.ndarray:
    # 각 에피소드의 시작 위치 = [0] + (마지막을 뺀 끝 위치들)
    # 예) ends=[161, 279, 400] → starts=[0, 161, 279]
    starts = np.concatenate(([0], episode_ends[:-1]))
    indices: list[int] = []
    for start, end in zip(starts, episode_ends, strict=True):  # (시작, 끝)을 한 쌍씩
        last_start = end - chunk_size  # 이 에피소드에서 묶음을 시작할 수 있는 마지막 시점
        if last_start < start:  # 에피소드가 K보다 짧으면 묶음을 하나도 못 만듦
            continue  # 이 에피소드는 건너뜀
        indices.extend(range(start, last_start + 1))  # start ~ last_start 를 전부 추가
    return np.asarray(indices, dtype=np.int64)


# ----------------------------------------------------------------------------
# PushtChunkDataset : PyTorch Dataset
#   입력 : states, actions, episode_ends, chunk_size, normalizer
#   출력 : dataset[i] → (state (5,), action_chunk (K, 2))
#   CIFAR-10의 (이미지, 라벨)과 같은 역할. 여기서는
#     이미지 → state (지금 상황), 라벨 → action_chunk (전문가가 이어서 한 K개 행동)
#   "슬라이딩 윈도우" : 시점을 한 칸씩 밀면서 묶음을 만들어서 샘플이 많아진다.
# ----------------------------------------------------------------------------
#   [복습] Dataset 규칙 : torch Dataset을 상속하면 함수 2개만 만들면 된다
#     __len__(self)        : 샘플이 총 몇 개인지
#     __getitem__(self, i) : i번째 샘플 하나를 반환
#   DataLoader는 이 두 함수만 써서 순서를 섞고(shuffle) 배치로 묶는다.
#   CIFAR-10 때 쓴 datasets.CIFAR10도 내부에 이 두 함수가 들어 있다.
# ----------------------------------------------------------------------------
class PushtChunkDataset(Dataset):
    """Dataset of (state, action_chunk) pairs using a sliding window."""

    def __init__(
        self,
        states: np.ndarray,  # (N, 5) 전체 상태
        actions: np.ndarray,  # (N, 2) 전체 행동
        episode_ends: np.ndarray,  # 에피소드 끝 위치들
        chunk_size: int,  # K, 한 묶음의 행동 개수 (기본 8)
        normalizer: Normalizer | None = None,  # None이면 정규화 안 함
    ) -> None:
        self.states = states
        self.actions = actions
        self.chunk_size = chunk_size
        self.normalizer = normalizer
        self.indices = build_valid_indices(episode_ends, chunk_size)  # 유효한 시작점 목록

    def __len__(self) -> int:
        # DataLoader가 "샘플이 몇 개냐"를 물을 때 호출. 유효한 시작점 개수 = 샘플 수
        return len(self.indices)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        # DataLoader가 idx번째 샘플을 달라고 할 때 호출
        t = int(self.indices[idx])  # idx번째 유효 시작점 → 실제 시점 t
        state = self.states[t]  # 시점 t의 상태, (5,)
        action_chunk = self.actions[t : t + self.chunk_size]  # t부터 K개 행동, (K, 2)

        if self.normalizer is not None:
            state = self.normalizer.normalize_state(state)
            action_chunk = self.normalizer.normalize_action(action_chunk)

        # numpy → torch 텐서, float32로. DataLoader가 이걸 배치로 쌓아
        # (B, 5), (B, K, 2) 모양을 만들어 준다.
        # [복습] torch.from_numpy : numpy 배열을 복사 없이 텐서로 바꿈 (메모리 공유)
        #        .float()         : float32로 맞춤. nn.Linear 가중치가 float32라 자료형이 다르면 에러
        #        정규화를 거치면 float64가 될 수 있어서 여기서 확실히 맞춰 준다
        return (
            torch.from_numpy(state).float(),
            torch.from_numpy(action_chunk).float(),
        )
