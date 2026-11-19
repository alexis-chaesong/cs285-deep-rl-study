"""Plot HW3 learning curves (eval return vs environment steps) from exp/*/log.csv."""

# ============================================================================
# plot_results.py : HW3 그래프 그리기
# ----------------------------------------------------------------------------
# 역할 : exp/ 안의 log.csv를 읽어 평가 점수 곡선(필요하면 Q값 곡선도)을 그린다.
# 입력 : exp/<환경>_<이름>_sd1_<날짜>/log.csv  (같은 이름이 여러 번이면 가장 최근 것)
# 출력 : plots/<그룹>.png
# 실행 : uv run --with matplotlib python plot_results.py [cartpole ...]
# ============================================================================

import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt

EXP = Path("exp")
OUT = Path("plots")

# 그룹 이름 → (실행 이름 접두어 목록, 제목, 목표 점수)
GROUPS = {
    "cartpole": (["CartPole-v1_dqn"], "DQN on CartPole-v1", 500),
    "lunarlander": (["LunarLander-v2_dqn"], "Double DQN on LunarLander-v2", 200),
    # 하이퍼파라미터 실험: 타깃 갱신 주기만 바꿈 (기본값 1000 = LunarLander-v2_dqn)
    "lunarlander_target_period": (
        ["LunarLander-v2_dqn_tup100", "LunarLander-v2_dqn_tup300", "LunarLander-v2_dqn", "LunarLander-v2_dqn_tup3000"],
        "LunarLander-v2: target update period (100 / 300 / 1000 / 3000)", 200),
}


def latest_run(prefix: str) -> Path:
    runs = sorted(EXP.glob(f"{prefix}_sd*"))
    if not runs:
        raise FileNotFoundError(f"no run for {prefix}")
    return runs[-1]


def read(run: Path, column: str) -> tuple[list[float], list[float]]:
    # 평가 줄과 학습 줄이 섞여 있어서, 해당 열이 비어 있지 않은 줄만 씀
    xs, ys = [], []
    with open(run / "log.csv") as f:
        for row in csv.DictReader(f):
            if row.get(column):
                xs.append(float(row["step"]))
                ys.append(float(row[column]))
    return xs, ys


def main() -> None:
    OUT.mkdir(exist_ok=True)
    for group in sys.argv[1:] or list(GROUPS):
        prefixes, title, target = GROUPS[group]
        fig, (ax, axq) = plt.subplots(1, 2, figsize=(11, 4))
        for prefix in prefixes:
            run = latest_run(prefix)
            xs, ys = read(run, "Eval_AverageReturn")
            ax.plot(xs, ys, label=prefix)
            qx, qy = read(run, "q_values")
            axq.plot(qx, qy, label=prefix)
            print(f"{prefix:30s} final {ys[-1]:7.1f}  best {max(ys):7.1f}  "
                  f"first step >= {target}: {next((int(x) for x, y in zip(xs, ys) if y >= target), '-')}")
        ax.axhline(target, ls="--", lw=1, color="gray")  # PDF의 목표 점수
        ax.set(xlabel="Environment steps", ylabel="Eval average return", title=title)
        axq.set(xlabel="Environment steps", ylabel="Mean Q value (batch)", title="Q values")
        for a in (ax, axq):
            a.grid(alpha=0.3)
            a.legend()
        fig.tight_layout()
        fig.savefig(OUT / f"{group}.png", dpi=150)
        print(f"saved plots/{group}.png")


if __name__ == "__main__":
    main()
