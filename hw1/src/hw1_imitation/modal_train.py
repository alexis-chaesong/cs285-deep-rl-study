# ============================================================================
# modal_train.py : 클라우드(Modal)에서 학습하는 파일 — 수강생용, 우리는 안 씀
# ----------------------------------------------------------------------------
# 역할 : Modal이라는 클라우드 GPU 서비스에 컨테이너를 만들어 train.py의 run_training을
#        원격으로 실행한다. README에 따르면 HW1은 노트북 CPU가 오히려 더 빠르다.
# 입력 : train.py와 같은 터미널 옵션
# 처리 : 의존성 설치된 이미지 생성 → 프로젝트 폴더 업로드 → T4 GPU에서 학습
# 출력 : Modal 볼륨(hw1-imitation-volume)에 기록 저장
# 실행 : uv run modal run src/hw1_imitation/modal_train.py   (Modal 계정 필요)
# 우리는 대신 로컬 GPU나 Colab을 쓰므로 이 파일은 읽기만 하면 된다.
# ============================================================================

from pathlib import Path

import modal  # 클라우드 실행 라이브러리

from hw1_imitation.train import TrainConfig, parse_train_config, run_training


APP_NAME = "hw1-imitation"
NETRC_PATH = Path("~/.netrc").expanduser()  # W&B 로그인 정보가 저장되는 파일
PROJECT_DIR = "/root/project"  # 컨테이너 안에서 프로젝트가 놓일 경로
VOLUME_PATH = "/vol"  # 결과를 저장할 영구 저장소 경로
DEFAULT_GPU = "T4"  # 사용할 GPU 종류
DEFAULT_CPU = 2.0  # CPU 코어 수
volume = modal.Volume.from_name("hw1-imitation-volume", create_if_missing=True)  # 없으면 새로 만듦


def load_gitignore_patterns() -> list[str]:
    """Translate .gitignore entries into Modal ignore globs."""
    # .gitignore에 적힌 파일(데이터, 기록 등)은 클라우드로 올리지 않도록 제외 목록을 만든다

    if not modal.is_local():  # 클라우드 안에서 실행 중이면 필요 없음
        return []

    root = Path(__file__).resolve().parents[2]  # 이 파일에서 2단계 위 = hw1/ 폴더
    gitignore_path = root / ".gitignore"
    if not gitignore_path.is_file():
        return []

    patterns: list[str] = []
    for line in gitignore_path.read_text(encoding="utf-8").splitlines():  # 한 줄씩
        entry = line.strip()
        if not entry or entry.startswith("#") or entry.startswith("!"):  # 빈 줄, 주석, 예외 규칙은 건너뜀
            continue
        entry = entry.lstrip("/")
        if entry.endswith("/"):  # 폴더 규칙이면 → 그 폴더 안 전부
            entry = entry.rstrip("/")
            patterns.append(f"**/{entry}/**")
        else:  # 파일 규칙이면 → 어느 위치에 있든 그 파일
            patterns.append(f"**/{entry}")
    return patterns


# Build a container image with the project's dependencies using uv.
# 컨테이너 이미지 = 클라우드에서 쓸 "설치가 끝난 컴퓨터" 설계도
image = modal.Image.debian_slim().apt_install("libgl1", "libglib2.0-0").uv_sync()  # 리눅스 + 그래픽 라이브러리 + 파이썬 패키지
if NETRC_PATH.is_file():  # W&B 로그인 정보가 있으면 같이 복사
    image = image.add_local_file(
        NETRC_PATH,
        remote_path="/root/.netrc",
        copy=True,
    )
image = image.add_local_dir(  # 현재 폴더(hw1)를 컨테이너로 복사 (gitignore 대상 제외)
    ".", remote_path=PROJECT_DIR, ignore=load_gitignore_patterns()
)


app = modal.App(APP_NAME)

env = {  # 컨테이너 안의 환경 변수
    "PYTHONPATH": f"{PROJECT_DIR}/src",  # hw1_imitation 패키지를 찾을 수 있게
    "WANDB_DIR": f"{VOLUME_PATH}/wandb",  # W&B 기록을 영구 저장소에
}


@app.function(  # 이 함수를 클라우드에서 실행하라는 표시
    volumes={VOLUME_PATH: volume},
    timeout=60 * 60 * 4,  # 최대 4시간
    env=env,
    image=image,
    gpu=DEFAULT_GPU,
    cpu=DEFAULT_CPU,
)
def train_remote(*args: str) -> None:
    defaults = TrainConfig()
    defaults.data_dir = Path(VOLUME_PATH) / "data"  # 데이터도 영구 저장소에 (다음 실행 때 재다운로드 안 함)
    config = parse_train_config(
        list(args),
        defaults=defaults,
        description="Train on Modal.",
    )
    run_training(config)  # train.py와 똑같은 학습
    volume.commit()  # 저장소 변경 내용 확정
