import argparse
import os
from datetime import datetime
import time

import gym
import numpy as np
import torch
import tqdm

from agents.pg_agent import PGAgent
from infrastructure import utils
from infrastructure import pytorch_util as ptu
from infrastructure.log_utils import setup_wandb, Logger, dump_log

MAX_NVIDEO = 2

# 이 코드는 앞서 살펴본 에이전트(PGAgent)와 데이터 수집 함수(utils)들을 하나로 엮어서, 
# 실제로 로봇이나 에이전트를 시뮬레이터 환경에서 굴리며 학습시키는 메인 실행 루프(Main Training Loop) 파일
# 5장에서 배운 "① 샘플 생성 ➔ ② 반환 추정 ➔ ③ 정책 개선"의 전체 사이클이 이 코드를 통해 반복 실행

# 1. 메인 학습 루프: run_training_loop
# 에이전트가 세상에 태어나서 반복적으로 경험을 쌓고, 실력을 키워나가는 가장 핵심적인 함수
def run_training_loop(logger, args):
    # set random seeds
    # 1. 재현성을 위한 시드(Seed) 고정 및 GPU 설정
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    ptu.init_gpu(use_gpu=not args.no_gpu, gpu_id=args.which_gpu)

    # make the gym environment
    # 2. Gym 시뮬레이터 환경 생성 및 기본 정보 추출
    env = gym.make(args.env_name, render_mode=None)
    discrete = isinstance(env.action_space, gym.spaces.Discrete) # 이산 행동인지 여부

    max_ep_len = args.ep_len or env.spec.max_episode_steps # 한 판의 최대 스텝 수 (지평, T)

    ob_dim = env.observation_space.shape[0] # 상태 차원 크기
    ac_dim = env.action_space.n if discrete else env.action_space.shape[0] # 행동 차원 크기
    # 핵심 의미: 실험의 기초 환경(예: CartPole 등)을 만들고, 상태와 행동의 크기가 몇 개인지를 파악하여 신경망을 만들 준비를 함 

    # simulation timestep, will be used for video saving
    if hasattr(env, "model"):
        fps = 1 / env.dt
    else:
        fps = env.env.metadata["render_fps"]

    # initialize agent
    # 3. 앞서 살펴본 PGAgent 에이전트 초기화 (하이퍼파라미터 주입)
    agent = PGAgent(
        ob_dim,
        ac_dim,
        discrete,
        n_layers=args.n_layers,
        layer_size=args.layer_size,
        gamma=args.discount,
        learning_rate=args.learning_rate,
        use_baseline=args.use_baseline,
        use_reward_to_go=args.use_reward_to_go,
        normalize_advantages=args.normalize_advantages,
        baseline_learning_rate=args.baseline_learning_rate,
        baseline_gradient_steps=args.baseline_gradient_steps,
        gae_lambda=args.gae_lambda,
    )
    # 핵심 의미: 6장과 7장에서 배운 옵션들(use_reward_to_go, 관측/행동 차원, 할인율 $\gamma$, GAE $\lambda$ 등)을 싹 모아서 에이전트 객체를 탄생시킴

    total_envsteps = 0
    start_time = time.time()

    # 4. 정해진 학습 반복 횟수(n_iter)만큼 루프 돌기
    for itr in range(args.n_iter):
        print(f"\n********** Iteration {itr} ************")
        # TODO: sample `args.batch_size` transitions using utils.sample_trajectories
        # make sure to use `max_ep_len`
        # [단계 ①]: 현재 정책으로 환경을 돌아다니며 데이터(Trajectory) 수집
        trajs, envsteps_this_batch = utils.sample_trajectories(
            env, agent.actor, args.batch_size, max_ep_len
        )
        total_envsteps += envsteps_this_batch

        # trajs should be a list of dictionaries of NumPy arrays, where each dictionary corresponds to a trajectory.
        # this line converts this into a single dictionary of lists of NumPy arrays.
        # 수집한 리스트 형태의 궤적을 에이전트가 다루기 좋은 딕셔너리로 변환
        trajs_dict = {k: [traj[k] for traj in trajs] for k in trajs[0]}

        # TODO: train the agent using the sampled trajectories and the agent's update function
        # [단계 ② & ③]: 수집한 데이터를 이용해 에이전트(액터 & 크리틱) 업데이트 수행
        train_info: dict = agent.update(
            trajs_dict["observation"],
            trajs_dict["action"],
            trajs_dict["reward"],
            trajs_dict["terminal"],
        )
        # 핵심 의미:utils.sample_trajectories를 통해 현재 정책 기준으로 시뮬레이터에서 데이터를 왕창 모음(5장 해부도의 ①번).
        # 모은 데이터를 agent.update에 던져주어 6장/7장의 수식대로 Q값/어드밴티지를 구하고 신경망 가중치를 고침 (해부도의 ②, ③번).

        # 5. 주기적으로 성능을 평가(Evaluation)하고 로그 기록하기
        if itr % args.scalar_log_freq == 0:
            # save eval metrics
            print("\nCollecting data for eval...")
            eval_trajs, eval_envsteps_this_batch = utils.sample_trajectories(
                env, agent.actor, args.eval_batch_size, max_ep_len
            )

            logs = utils.compute_metrics(trajs, eval_trajs)
            # compute additional metrics
            logs.update(train_info)
            logs["Train_EnvstepsSoFar"] = total_envsteps
            logs["TimeSinceStart"] = time.time() - start_time
            if itr == 0:
                logs["Initial_DataCollection_AverageReturn"] = logs[
                    "Train_AverageReturn"
                ]

            # perform the logging
            for key, value in logs.items():
                print("{} : {}".format(key, value))
            logger.log(logs, itr)       # 텐서보드나 Wandb에 기록
            print("Done logging...\n\n", flush=True)
            # 핵심 의미: 학습이 잘 되고 있는지 확인하기 위해, 탐험용 노이즈를 끄거나 순수하게 평가용 데이터(eval_trajs)를 뽑아 평균 보상(Eval_AverageReturn)을 측정하고 기록

        # 6. 비디오 로깅 설정이 되어 있다면 에이전트의 플레이 영상을 파일로 저장
        if args.video_log_freq != -1 and itr % args.video_log_freq == 0:
            print("\nCollecting video rollouts...")
            eval_video_trajs = utils.sample_n_trajectories(
                env, agent.actor, MAX_NVIDEO, max_ep_len, render=True
            )

            logger.log_trajs_as_videos(
                eval_video_trajs,
                itr,
                fps=fps,
                max_videos_to_save=MAX_NVIDEO,
                video_title="eval_rollouts",
            )

    # 학습이 모두 끝나면 최종 모델과 로그 저장
    dump_log(agent, logger, args, args.save_dir)

# 2. 하이퍼파라미터 정의: setup_arguments
# 터미널이나 명령어 창에서 이 코드를 실행할 때 조절할 수 있는 다양한 설정값(인자)들을 정의하는 곳
# 6~7장에서 배운 핵심 개념들이 옵션으로 그대로 녹아 있음 
def setup_arguments(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_name", type=str, default='CartPole-v0') # 학습할 환경 이름 (예: CartPole-v0)
    parser.add_argument("--exp_name", type=str, default='exp')
    parser.add_argument("--n_iter", "-n", type=int, default=200)

    parser.add_argument("--use_reward_to_go", "-rtg", action="store_true")  # 6장 트릭 ①인 Reward-to-go를 쓸 것인가? (켜는 것이 표준)
    parser.add_argument("--use_baseline", action="store_true")      # 6장/7장의 기준선(Baseline / 크리틱)을 쓸 것인가?
    parser.add_argument("--baseline_learning_rate", "-blr", type=float, default=5e-3)
    parser.add_argument("--baseline_gradient_steps", "-bgs", type=int, default=5)
    parser.add_argument("--gae_lambda", type=float, default=None)       # 7장의 GAE $\lambda$ 파라미터. (값에 따라 편향과 분산이 조절됨)
    parser.add_argument("--normalize_advantages", "-na", action="store_true")       # 어드밴티지 값들을 평균 0, 표준편차 1로 정규화할 것인가? (학습 안정성에 큰 도움을 줌)
    parser.add_argument(
        "--batch_size", "-b", type=int, default=1000    # 한 번 정책을 업데이트할 때 모을 총 스텝 수.
    )  # steps collected per train iteration
    parser.add_argument(
        "--eval_batch_size", "-eb", type=int, default=400
    )  # steps collected per eval iteration

    parser.add_argument("--discount", type=float, default=1.0)
    parser.add_argument("--learning_rate", "-lr", type=float, default=5e-3)
    parser.add_argument("--n_layers", "-l", type=int, default=2)
    parser.add_argument("--layer_size", "-s", type=int, default=64)

    parser.add_argument(
        "--ep_len", type=int
    )  # students shouldn't change this away from env's default
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--no_gpu", "-ngpu", action="store_true")
    parser.add_argument("--which_gpu", "-gpu_id", default=0)
    parser.add_argument("--video_log_freq", type=int, default=-1)
    parser.add_argument("--scalar_log_freq", type=int, default=1)

    args = parser.parse_args(args=args)

    return args

# 3. 프로그램 진입점: main
def main(args):
    # Create directory for logging
    logdir_prefix = "exp"  # Keep for autograder

    exp_name = f"{args.env_name}_{args.exp_name}_sd{args.seed}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    config = vars(args)
    setup_wandb(project='cs285_hw2', name=exp_name, config=config)  # 실험 도구(Wandb) 연동
    args.save_dir = os.path.join(logdir_prefix, exp_name)
    os.makedirs(args.save_dir, exist_ok=True)
    logger = Logger(os.path.join(args.save_dir, 'log.csv'))         # CSV 로그 저장 폴더 생성

    run_training_loop(logger, args)         # 학습 루프 실행!


if __name__ == "__main__":
    args = setup_arguments()
    main(args)

    # 핵심 의미: 실험 결과를 기록할 폴더를 만들고, 실험 관리 도구인 Wandb(Weights & Biases)를 세팅한 뒤, 앞서 본 run_training_loop를 호출하여 전체 강화학습 파이프라인을 뫼시는 관문