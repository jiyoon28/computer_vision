"""
split CSV 저장 / 로드 헬퍼

CSV에는 REPO_ROOT 기준 상대경로만 저장한다.

이유:
절대경로를 저장하면 다른 PC나 다른 위치로 프로젝트를 옮겼을 때
CSV 안의 경로가 전부 깨진다.

    저장: save_split(df, "train")   → 절대경로를 상대경로로 변환해서 저장
    로드: load_split("train")       → 상대경로를 다시 절대경로로 복원
"""

from pathlib import Path

import pandas as pd

from config import (
    REPO_ROOT,
    SPLIT_DIR,
)


# CSV에서 경로가 들어있는 컬럼
PATH_COLUMNS = [
    "image_path",
    "mask_path",
]


def to_relative(path):
    """
    절대경로 → REPO_ROOT 기준 상대경로

    mask가 없는 정상 이미지는 빈 문자열이므로 그대로 통과시킨다.
    """

    if not path:
        return ""

    return Path(path).relative_to(REPO_ROOT).as_posix()


def to_absolute(path):
    """
    REPO_ROOT 기준 상대경로 → 절대경로
    """

    if not path:
        return ""

    return str(REPO_ROOT / path)


def save_split(df, name):
    """
    DataFrame을 SPLIT_DIR/{name}.csv 로 저장

    경로 컬럼은 상대경로로 변환해서 저장한다.
    """

    # 원본 DataFrame을 건드리지 않기 위해 복사
    out = df.copy()

    for column in PATH_COLUMNS:

        if column in out.columns:
            out[column] = out[column].map(to_relative)

    csv_path = SPLIT_DIR / f"{name}.csv"

    out.to_csv(
        csv_path,
        index=False
    )

    return csv_path


def resolve_paths(df):
    """
    DataFrame의 경로 컬럼을 절대경로로 복원

    CSV를 직접 read_csv로 읽었을 때 사용한다.
    복원하지 않으면 상대경로 그대로라서
    실행 위치(cwd)에 따라 FileNotFoundError가 난다.
    """

    out = df.copy()

    for column in PATH_COLUMNS:

        if column in out.columns:
            out[column] = out[column].map(to_absolute)

    return out


def read_csv(csv_path):
    """
    split CSV 한 개를 읽어서 절대경로로 복원된 DataFrame 반환
    """

    # mask_path가 비어 있는 행을 NaN이 아닌 빈 문자열로 읽기 위해
    # keep_default_na=False 사용
    df = pd.read_csv(
        csv_path,
        keep_default_na=False
    )

    return resolve_paths(df)


def load_split(name):
    """
    SPLIT_DIR/{name}.csv 를 읽어서 DataFrame 반환

    경로 컬럼은 절대경로로 복원해서 돌려준다.
    바로 PIL / cv2로 열 수 있는 상태.
    """

    return read_csv(SPLIT_DIR / f"{name}.csv")
