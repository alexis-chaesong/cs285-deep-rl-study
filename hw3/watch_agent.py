"""Load a trained DQN agent from exp/ and save rollout videos as mp4 files."""

# ============================================================================
# watch_agent.py : 학습된 DQN 에이전트가 실제로 어떻게 움직이는지 영상으로 저장
# ----------------------------------------------------------------------------
# 역할 : exp/<실행>/agent.pt(학습된 가중치)를 불러와 ε=0(평가 모드)으로 한 판씩 돌리고,
#        매 스텝 화면(rgb_array)을 모아 mp4로 저장한다. W&B 없이 내 컴퓨터에서 바로 볼 수 있음.
# 입력 : 설정 yaml (예: experiments/dqn/cartpole.yaml), 실행 폴더 (생략하면 가장 최근 것)
# 출력 : videos/<실행 폴더 이름>_ep<번호>.mp4
# 실행 : uv run python watch_agent.py -cfg experiments/dqn/cartpole.yaml [--run exp/...] [-n 2]
# ============================================================================

import argparse
import sys
from pathlib import Path

import imageio
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent / "src"))
from agents.dqn_agent import DQNAgent  # noqa: E402
from infrastructure import pytorch_util as ptu  # noqa: E402
from scripts.run_dqn import make_config  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", "-cfg", required=True)
    parser.add_argument("--run", default=None, help="exp/ 아래 실행 폴더. 생략하면 log_name이 같은 가장 최근 것")
    parser.add_argument("--episodes", "-n", type=int, default=2)
    parser.add_argument("--fps", type=int, default=30)
    args = parser.parse_args()

    ptu.init_gpu(use_gpu=False)  # 영상 하나 뽑는 데는 CPU로 충분
    config = make_config(args.config_file)
    run = Path(args.run) if args.run else sorted(Path("exp").glob(f"{config['log_name']}_sd*"))[-1]

    env = config["make_env"](eval=True, render=True)
    agent = DQNAgent(env.observation_space.shape, env.action_space.n, **config["agent_kwargs"])
    agent.load_state_dict(torch.load(run / "agent.pt", map_location="cpu"))
    agent.eval()

    out_dir = Path("videos")
    out_dir.mkdir(exist_ok=True)
    for ep in range(args.episodes):
        obs = np.asarray(env.reset())
        frames, total, done = [], 0.0, False
        while not done:
            action = agent.get_action(obs)  # epsilon 기본값 0 = 평가 모드(항상 Q 최고점 행동)
            obs, reward, done, info = env.step(action)
            obs = np.asarray(obs)
            total += reward
        # 이 gym 버전은 render()가 "지난 호출 이후 쌓인 프레임 목록"을 돌려줘서, 판이 끝난 뒤 한 번에 받음
        frames = env.render()
        path = out_dir / f"{run.name}_ep{ep}.mp4"
        imageio.mimsave(path, frames, fps=args.fps)
        print(f"episode {ep}: return {total:.1f}, {len(frames)} frames -> {path}")


if __name__ == "__main__":
    main()
