from collections import OrderedDict
import numpy as np
import copy
from networks.policies import MLPPolicy
import gym
import cv2
from infrastructure import pytorch_util as ptu
from typing import Dict, Tuple, List

# 이 코드는 5장의 "모든 RL 알고리즘의 해부도" 중 ① 샘플 생성(정책 실행) 단계와 데이터를 정리하는 과정을 담당하는 핵심 인프라 코드
# 로봇이나 에이전트가 환경(Gym 시뮬레이터) 속에서 직접 부딪히며 경험(Trajectory, 궤적)을 수집하고, 이를 학습에 쓰기 좋게 가공하는 함수들이 모여 있음 
############################################
############################################

# 1. 단일 궤적 수집기: sample_trajectory
# 에이전트가 환경 속에서 한 판(Episode)을 처음부터 끝까지 수행하며 기록장을 쓰는 함수
def sample_trajectory(
    env: gym.Env, policy: MLPPolicy, max_length: int, render: bool = False
) -> Dict[str, np.ndarray]:
    """Sample a rollout in the environment from a policy."""
    ob = env.reset()            # 1. 환경을 초기화하고 초기 상태(Observation)를 받음
    obs, acs, rewards, next_obs, terminals, image_obs = [], [], [], [], [], []
    steps = 0
    while True:
        # 화면 렌더링 옵션이 켜져 있으면 현재 화면을 이미지로 캡처함 (시각화/비디오 저장용)
        # render an image
        if render:
            if hasattr(env, "sim"):
                img = env.sim.render(camera_name="track", height=500, width=500)[::-1]
            else:
                img = env.render(mode="single_rgb_array")
            image_obs.append(
                cv2.resize(img, dsize=(250, 250), interpolation=cv2.INTER_CUBIC)
            )

        # TODO use the most recent ob to decide what to do
        # 2. 현재 관측값(ob)을 정책 신경망에 넣어 다음에 할 행동(ac)을 결정함
        ac = policy.get_action(ob)

        # TODO: take that action and get reward and next ob
        # 3. 환경에 행동을 실제로 취하고, 다음 상태와 보상, 종료 여부(done)를 받아옴
        next_ob, rew, done, info = env.step(ac)

        # TODO rollout can end due to done, or due to max_length
        # 4. 에피소드가 끝났거나, 최대 허용 길이(max_length)에 도달했는지 체크
        steps += 1
        rollout_done = done or steps >= max_length

        # record result of taking that action
        # 5. 이번 스텝에서 일어난 일들을 각각의 리스트에 기록함
        obs.append(ob)
        acs.append(ac)
        rewards.append(rew)
        next_obs.append(next_ob)
        terminals.append(rollout_done)

        ob = next_ob  # jump to next timestep # 다음 스텝을 위해 상태를 업데이트함

        # end the rollout if the rollout ended
        # 한 판이 끝났으면 루프 탈출
        if rollout_done:
            break

    # 6. 수집한 리스트들을 신경망 학습에 쓰기 좋은 NumPy 배열로 변환해서 딕셔너리로 반환
    return {
        "observation": np.array(obs, dtype=np.float32),
        "image_obs": np.array(image_obs, dtype=np.uint8),
        "reward": np.array(rewards, dtype=np.float32),
        "action": np.array(acs, dtype=np.float32),
        "next_observation": np.array(next_obs, dtype=np.float32),
        "terminal": np.array(terminals, dtype=np.float32),
    }

# 핵심 의미: 5장에서 배운 env.step(action)을 반복하며 로봇이 큐브를 잡고 움직이는 등의 상호작용 기록(observation, action, reward 등)을 한 판 단위로 예쁘게 포장하는 과정

# 2. 배치 단위 샘플 수집기: sample_trajectories
# 정책 경사(Policy Gradient) 같은 On-policy 알고리즘은 한 판만으로는 데이터가 부족함
# "충분한 양의 스텝 수"가 쌓일 때까지 여러 판을 연달아 수집하는 함수

def sample_trajectories(
    env: gym.Env,
    policy: MLPPolicy,
    min_timesteps_per_batch: int,
    max_length: int,
    render: bool = False,
) -> Tuple[List[Dict[str, np.ndarray]], int]:
    """Collect rollouts using policy until we have collected min_timesteps_per_batch steps."""
    timesteps_this_batch = 0
    trajs = []

    # 목표한 최소 스텝 수(min_timesteps_per_batch)를 채울 때까지 반복
    while timesteps_this_batch < min_timesteps_per_batch:
        # collect rollout
        traj = sample_trajectory(env, policy, max_length, render) # 한 판 돌기
        trajs.append(traj) # 수집한 판을 리스트에 추가

        # count steps
        timesteps_this_batch += get_traj_length(traj) # 총 스텝 수 누적
    return trajs, timesteps_this_batch

# 핵심 의미: "이번 학습에 최소 5000스텝 분량의 데이터가 필요해!" 하면 이 함수가 알아서 에피소드 여러 개를 돌며 5000스텝이 넘을 때까지 데이터를 긁어옴 

# 3. 정해진 횟수만큼 수집하기: sample_n_trajectories
def sample_n_trajectories(
    env: gym.Env, policy: MLPPolicy, ntraj: int, max_length: int, render: bool = False
):
    """Collect ntraj rollouts."""
    trajs = []
    for _ in range(ntraj):      # 정확히 지정된 판 수(ntraj)만큼만 반복
        # collect rollout
        traj = sample_trajectory(env, policy, max_length, render)
        trajs.append(traj)
    return trajs

# 핵심 의미: 학습 중간중간 성능을 평가(Evaluation)할 때 주로 사용. "딱 10판만 돌려보고 평균 점수가 얼마나 나오는지 보자" 할 때 쓰임. 

# 4. 로그 및 성능 측정기: compute_metrics
# 학습이 잘 되고 있는지 텐서보드(TensorBoard)나 콘솔에 기록하기 위해 각종 통계 지표를 계산
def compute_metrics(trajs, eval_trajs):
    """Compute metrics for logging."""

    # returns, for logging
    # 1. 훈련 데이터와 평가 데이터의 에피소드별 총 보상(Returns) 계산
    train_returns = [traj["reward"].sum() for traj in trajs]
    eval_returns = [eval_traj["reward"].sum() for eval_traj in eval_trajs]

    # 2. 에피소드 길이(몇 스텝 동안 살아남았는지) 계산
    # episode lengths, for logging
    train_ep_lens = [len(traj["reward"]) for traj in trajs]
    eval_ep_lens = [len(eval_traj["reward"]) for eval_traj in eval_trajs]

    # 3. 로그 딕셔너리에 평균, 표준편차, 최댓값, 최솟값 등을 보기 좋게 정리
    # decide what to log
    logs = OrderedDict()
    logs["Eval_AverageReturn"] = np.mean(eval_returns)  # 평가 평균 보상 (성능 지표!)
    logs["Eval_StdReturn"] = np.std(eval_returns)
    logs["Eval_MaxReturn"] = np.max(eval_returns)
    logs["Eval_MinReturn"] = np.min(eval_returns)
    logs["Eval_AverageEpLen"] = np.mean(eval_ep_lens)

    logs["Train_AverageReturn"] = np.mean(train_returns) # 훈련 평균 보상
    logs["Train_StdReturn"] = np.std(train_returns)
    logs["Train_MaxReturn"] = np.max(train_returns)
    logs["Train_MinReturn"] = np.min(train_returns)
    logs["Train_AverageEpLen"] = np.mean(train_ep_lens)

    return logs

# 핵심 의미: 6장 퀴즈에서 강조했듯, 강화학습에서는 가짜 Loss 값 대신 Eval_AverageReturn(평균 보상)이나 성공률을 봐야 함.
# 이 함수가 바로 그 평가 지표들을 예쁘게 묶어주는 역할을 함 

# 5. 데이터 형태 변환기: convert_listofrollouts와 get_traj_length
def convert_listofrollouts(trajs):
    """
    Take a list of rollout dictionaries and return separate arrays, where each array is a concatenation of that array
    from across the rollouts.
    """
    # 여러 판의 궤적(딕셔너리 리스트)을 받아 각각 하나의 거대한 1차원 배열로 이어붙임
    observations = np.concatenate([traj["observation"] for traj in trajs])
    actions = np.concatenate([traj["action"] for traj in trajs])
    next_observations = np.concatenate([traj["next_observation"] for traj in trajs])
    terminals = np.concatenate([traj["terminal"] for traj in trajs])
    concatenated_rewards = np.concatenate([traj["reward"] for traj in trajs])
    unconcatenated_rewards = [traj["reward"] for traj in trajs]     # 판별로 쪼개진 보상 리스트 유지
    return (
        observations,
        actions,
        next_observations,
        terminals,
        concatenated_rewards,
        unconcatenated_rewards,
    )

# 핵심 의미: 앞서 첫 번째 코드의 update 함수에서 본 np.concatenate와 완벽히 이어지는 부분!
# 환경에서 판별로 수집된 날것의 경험 리스트를 파이토치 신경망에 한 번에 집어넣을 수 있도록 거대한 일렬 데이터로 쫙 펴주는(Flatten) 전처리 도구

def get_traj_length(traj):
    return len(traj["reward"])  # 궤적의 길이(스텝 수) 반환

'''
이 코드를 통해 "시뮬레이터 환경에서 에이전트가 어떻게 데이터를 수집하고($sample\_trajectory$), 
그걸 모아서 신경망 학습에 쓰기 좋은 형태($convert\_listofrollouts$)로 바꾸는지" 
이 수집된 데이터가 바로 앞서 보았던 PGAgent.update() 함수로 곧바로 투입되는 것임.!
'''