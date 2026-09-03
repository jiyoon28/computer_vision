import pandas as pd

from sklearn.model_selection import train_test_split

from config import (
    RAW_DATA_DIR,
    SPLIT_DIR,
    SEED,
)

from splits import save_split


# ============================================================
# 1. Image / Mask Pair 찾기
# ============================================================

def collect_pairs():

    # 모든 이미지 정보를 저장할 list
    rows = []

    # 예:
    # MT_Blowhole
    # MT_Break
    # MT_Crack
    # MT_Fray
    # MT_Free
    # MT_Uneven
    class_dirs = sorted(
        RAW_DATA_DIR.glob("MT_*")
    )


    # 각 defect class 폴더를 반복
    for class_dir in class_dirs:

        # 예:
        # MT_Crack → Crack
        defect_type = class_dir.name.replace(
            "MT_",
            ""
        )


        # 실제 이미지들이 들어 있는 폴더
        image_dir = class_dir / "Imgs"


        # MT_Free는 정상 이미지
        #
        # Free:
        # is_defect = 0
        #
        # 나머지:
        # is_defect = 1
        is_defect = int(
            defect_type.lower() != "free"
        )


        # 모든 jpg 이미지 확인
        for image_path in sorted(
            image_dir.glob("*.jpg")
        ):

            # 같은 이름의 png를 Ground Truth Mask로 사용
            #
            # exp001.jpg
            # ↓
            # exp001.png
            mask_path = image_path.with_suffix(
                ".png"
            )


            # defect 이미지라면 mask가 반드시 존재해야 한다.
            if is_defect == 1 and not mask_path.exists():

                raise FileNotFoundError(
                    f"Ground Truth Mask를 찾을 수 없습니다:\n"
                    f"{mask_path}"
                )


            # 한 이미지에 대한 정보를 dictionary 형태로 저장
            rows.append(
                {
                    "image_path": str(image_path),

                    # 정상 이미지의 경우 mask가 없는 경우도 있으므로
                    # 빈 문자열로 저장
                    "mask_path": (
                        str(mask_path)
                        if mask_path.exists()
                        else ""
                    ),

                    "defect_type": defect_type,

                    "is_defect": is_defect,
                }
            )


    # list → pandas DataFrame
    df = pd.DataFrame(rows)

    return df


# ============================================================
# 2. Normal Image 수 조정
# ============================================================

def balance_free_images(df):

    # defect 이미지만 선택
    defect_df = df[
        df["is_defect"] == 1
    ].copy()


    # 정상(Free) 이미지만 선택
    free_df = df[
        df["is_defect"] == 0
    ].copy()


    print()
    print("Defect images :", len(defect_df))
    print("Normal images :", len(free_df))


    # Magnetic Tile Dataset에는 Normal 이미지가 훨씬 많다.
    #
    # 첫 실습에서는 image-level imbalance를 줄이기 위해
    # normal 수를 defect 전체 이미지 수 정도로 제한한다.
    n_free = min(
        len(free_df),
        len(defect_df)
    )


    # 정상 이미지 중 일부를 random sampling
    free_df = free_df.sample(
        n=n_free,
        random_state=SEED
    )


    # defect + normal 합치기
    balanced_df = pd.concat(
        [
            defect_df,
            free_df
        ],
        ignore_index=True
    )


    # 전체 데이터 순서를 랜덤하게 섞기
    balanced_df = balanced_df.sample(
        frac=1,
        random_state=SEED
    ).reset_index(
        drop=True
    )


    return balanced_df


# ============================================================
# 3. Train / Validation / Test Split
# ============================================================

def split_dataset(df):

    # 전체 데이터:
    #
    # Train 70%
    # Validation 15%
    # Test 15%


    # 먼저:
    #
    # Train = 70%
    # Temp  = 30%
    train_df, temp_df = train_test_split(
        df,

        test_size=0.30,

        random_state=SEED,

        # defect type의 비율을
        # train / temp에서 비슷하게 유지
        stratify=df["defect_type"],
    )


    # Temp 30%를 반으로 나눠서:
    #
    # Validation = 15%
    # Test       = 15%
    val_df, test_df = train_test_split(
        temp_df,

        test_size=0.50,

        random_state=SEED,

        stratify=temp_df["defect_type"],
    )


    return (
        train_df,
        val_df,
        test_df,
    )


# ============================================================
# 4. Main
# ============================================================

def main():

    # --------------------------------
    # Image / Mask pair 수집
    # --------------------------------

    df = collect_pairs()


    print()
    print("====================================")
    print("Original Dataset")
    print("====================================")

    print(
        df["defect_type"].value_counts()
    )


    # --------------------------------
    # Normal image balancing
    # --------------------------------

    df = balance_free_images(df)


    print()
    print("====================================")
    print("After Balancing")
    print("====================================")

    print(
        df["defect_type"].value_counts()
    )


    # --------------------------------
    # Train / Val / Test split
    # --------------------------------

    train_df, val_df, test_df = (
        split_dataset(df)
    )


    # --------------------------------
    # CSV 저장
    # --------------------------------

    # save_split()이 image_path / mask_path를
    # REPO_ROOT 기준 상대경로로 바꿔서 저장한다.
    save_split(train_df, "train")
    save_split(val_df, "val")
    save_split(test_df, "test")


    print()
    print("====================================")
    print("Dataset Split")
    print("====================================")

    print(
        f"Train : {len(train_df)}"
    )

    print(
        f"Val   : {len(val_df)}"
    )

    print(
        f"Test  : {len(test_df)}"
    )


    print()
    print("CSV 파일 저장 완료:")
    print(SPLIT_DIR)


# 이 파일을 직접 실행했을 때만 main 실행
if __name__ == "__main__":
    main()