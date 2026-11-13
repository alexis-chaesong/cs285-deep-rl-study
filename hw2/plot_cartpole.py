"""Plot Experiment 1 (CartPole) learning curves: small batch and large batch."""

# ============================================================================
# plot_cartpole.py : HW2 실험 1 그래프 그리기
# ----------------------------------------------------------------------------
# 역할 : exp/ 안의 CartPole 실행 8개의 log.csv를 읽어 학습 곡선 2장을 그린다.
#        x축은 PDF 요구대로 반복 횟수가 아니라 환경 스텝 수(Train_EnvstepsSoFar).
# 입력 : exp/CartPole-v0_<이름>_sd1_<날짜>/log.csv  (같은 이름이 여러 번이면 가장 최근 것)
# 출력 : plots/cartpole_small_batch.png, plots/cartpole_large_batch.png
# 실행 : uv run --with matplotlib python plot_cartpole.py
# ============================================================================

import csv
from pathlib import Path

import matplotlib.pyplot as plt

EXP = Path("exp")
GROUPS = {
    "small_batch": ["cartpole", "cartpole_rtg", "cartpole_na", "cartpole_rtg_na"],
    "large_batch": ["cartpole_lb", "cartpole_lb_rtg", "cartpole_lb_na", "cartpole_lb_rtg_na"],
}


def latest_run(name: str) -> Path:
    # "CartPole-v0_cartpole_sd1_..."이 "cartpole_rtg"와 헷갈리지 않도록 "_sd"까지 붙여서 찾음
    runs = sorted(EXP.glob(f"CartPole-v0_{name}_sd*"))
    if not runs:
        raise FileNotFoundError(f"no run for {name}")
    return runs[-1]


def read_curve(run: Path) -> tuple[list[float], list[float]]:
    xs, ys = [], []
    with open(run / "log.csv") as f:
        for row in csv.DictReader(f):
            xs.append(float(row["Train_EnvstepsSoFar"]))
            ys.append(float(row["Eval_AverageReturn"]))
    return xs, ys


def main() -> None:
    out_dir = Path("plots")
    out_dir.mkdir(exist_ok=True)
    for group, names in GROUPS.items():
        fig, ax = plt.subplots(figsize=(7, 4))
        for name in names:
            xs, ys = read_curve(latest_run(name))
            ax.plot(xs, ys, label=name)
            print(f"{name:20s} final {ys[-1]:6.1f}  best {max(ys):6.1f}  "
                  f"first step reaching 200: {next((int(x) for x, y in zip(xs, ys) if y >= 200), '-')}")
        ax.axhline(200, ls="--", lw=1, color="gray")  # CartPole 최고 점수
        ax.set(xlabel="Environment steps (Train_EnvstepsSoFar)", ylabel="Eval average return",
               title=f"CartPole-v0, {group.replace('_', ' ')} (batch {1000 if group == 'small_batch' else 4000})",
               ylim=(0, 210))
        ax.grid(alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(out_dir / f"cartpole_{group}.png", dpi=150)
        print(f"saved plots/cartpole_{group}.png\n")


if __name__ == "__main__":
    main()
