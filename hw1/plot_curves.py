"""Plot training curves (loss and eval reward) for the best MSE and flow runs."""

# ============================================================================
# plot_curves.py : 보고서용 학습 곡선 그리기
# ----------------------------------------------------------------------------
# 역할 : exp/mse, exp/flow 실행의 손실과 평가 점수를 W&B에서 받아 matplotlib로 그린다.
#        (log.csv에는 평가 점수가 빠져 있어서 W&B 기록을 쓴다)
# 입력 : exp/<이름>/wandb/run-<id>.wandb 파일 이름에서 W&B 실행 id를 읽음
# 출력 : plots/training_curves.png (왼쪽: 스텝 vs 손실, 오른쪽: 스텝 vs 점수)
# 실행 : uv run --with matplotlib python plot_curves.py
#        (--with : pyproject.toml을 건드리지 않고 이번 실행에만 matplotlib을 추가)
# ============================================================================

from pathlib import Path

import matplotlib.pyplot as plt
import wandb

PROJECT = "hw1-imitation"
RUNS = {"MSE": Path("exp/mse"), "Flow matching": Path("exp/flow")}  # 범례 이름 → 실행 폴더
TARGETS = {"MSE": 0.5, "Flow matching": 0.7}  # 과제 목표 점수 (점선으로 표시)


def find_run_id(exp_dir: Path) -> str:
    # exp/mse/wandb/run-dej2mkhb.wandb → "dej2mkhb"
    files = list((exp_dir / "wandb").glob("run-*.wandb"))
    if not files:
        raise FileNotFoundError(f"No W&B run file in {exp_dir / 'wandb'}")
    return files[0].stem.removeprefix("run-")


def main() -> None:
    api = wandb.Api()  # wandb login 정보로 접속
    fig, (ax_loss, ax_rew) = plt.subplots(1, 2, figsize=(11, 4))

    for label, exp_dir in RUNS.items():
        run = api.run(f"{api.default_entity}/{PROJECT}/{find_run_id(exp_dir)}")
        # samples를 크게 줘야 W&B가 점을 솎아 내지 않고 전부 돌려줌
        loss = run.history(keys=["train/loss"], pandas=False, samples=100_000)
        rew = run.history(keys=["eval/mean_reward"], pandas=False, samples=1_000)

        line, = ax_loss.plot(
            [r["_step"] for r in loss], [r["train/loss"] for r in loss], label=label, alpha=0.8
        )
        ax_rew.plot(
            [r["_step"] for r in rew], [r["eval/mean_reward"] for r in rew],
            marker="o", label=label, color=line.get_color(),
        )
        ax_rew.axhline(TARGETS[label], ls="--", lw=1, color=line.get_color(), alpha=0.6)  # 목표선
        print(f"{label}: final reward {rew[-1]['eval/mean_reward']:.3f}, "
              f"best {max(r['eval/mean_reward'] for r in rew):.3f}")

    # 손실은 0.01~1 범위를 오가므로 로그 축이 보기 편함
    # (두 정책의 손실은 맞히는 대상이 달라서 값 자체는 비교하지 않는다)
    ax_loss.set(xlabel="Training step", ylabel="Train loss", yscale="log", title="Training loss")
    ax_rew.set(xlabel="Training step", ylabel="Mean eval reward (100 episodes)",
               ylim=(0, 1), title="Evaluation reward (dashed: target)")
    for ax in (ax_loss, ax_rew):
        ax.grid(alpha=0.3)
        ax.legend()

    out = Path("plots") / "training_curves.png"
    out.parent.mkdir(exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
