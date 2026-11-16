"""Plot Experiment 2 (HalfCheetah baseline) and Experiment 3 (LunarLander GAE) learning curves."""

# ============================================================================
# plot_experiments.py : HW2 실험 2, 3 그래프 그리기
# ----------------------------------------------------------------------------
# 역할 : exp/ 안의 log.csv를 읽어 학습 곡선을 그린다. (plot_cartpole.py와 같은 방식)
# 입력 : exp/<환경>_<이름>_sd1_<날짜>/log.csv  (같은 이름이 여러 번이면 가장 최근 것)
# 출력 : plots/cheetah_eval.png, plots/cheetah_baseline_loss.png, plots/lunar_lander_gae.png
# 실행 : uv run --with matplotlib python plot_experiments.py [cheetah|lunar]
# ============================================================================

import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt

EXP = Path("exp")
OUT = Path("plots")


def latest_run(env: str, name: str) -> Path:
    # "_sd"까지 붙여서 "cheetah"가 "cheetah_baseline"과 헷갈리지 않게 함
    runs = sorted(EXP.glob(f"{env}_{name}_sd*"))
    if not runs:
        raise FileNotFoundError(f"no run for {env} {name}")
    return runs[-1]


def read_column(run: Path, column: str) -> tuple[list[float], list[float]]:
    xs, ys = [], []
    with open(run / "log.csv") as f:
        for row in csv.DictReader(f):
            xs.append(float(row["Train_EnvstepsSoFar"]))
            ys.append(float(row[column]))
    return xs, ys


def plot(env: str, names: list[str], column: str, title: str, out: str, target: float | None = None) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    for name in names:
        xs, ys = read_column(latest_run(env, name), column)
        ax.plot(xs, ys, label=name)
        print(f"{name:28s} {column}: final {ys[-1]:8.1f}  best {max(ys):8.1f}")
    if target is not None:
        ax.axhline(target, ls="--", lw=1, color="gray")  # PDF의 목표 점수
    ax.set(xlabel="Environment steps (Train_EnvstepsSoFar)", ylabel=column, title=title)
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / out, dpi=150)
    print(f"saved plots/{out}\n")


def main() -> None:
    OUT.mkdir(exist_ok=True)
    which = sys.argv[1:] or ["cheetah", "lunar"]
    if "cheetah" in which:
        env = "HalfCheetah-v4"
        plot(env, ["cheetah", "cheetah_baseline", "cheetah_baseline_bgs2"], "Eval_AverageReturn",
             "HalfCheetah-v4: with and without a baseline", "cheetah_eval.png", target=300)
        # 크리틱 손실은 기준선이 있는 두 실행에만 있음
        plot(env, ["cheetah_baseline", "cheetah_baseline_bgs2"], "Baseline Loss",
             "HalfCheetah-v4: critic loss, -bgs 5 vs 2", "cheetah_baseline_loss.png")
    if "lunar" in which:
        names = [f"lunar_lander_lambda{lam}" for lam in ["0", "0.95", "0.98", "0.99", "1"]]
        plot("LunarLander-v2", names, "Eval_AverageReturn",
             "LunarLander-v2: GAE lambda", "lunar_lander_gae.png", target=150)


if __name__ == "__main__":
    main()
