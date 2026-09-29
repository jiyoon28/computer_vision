# YOLO26 Semantic Segmentation — Magnetic Tile

기존 U-Net과 동일한 train/val/test CSV를 사용해 YOLO26-sem을 학습합니다.
`yolo26n-sem.pt`는 semantic, `yolo26n-seg.pt`는 instance 모델입니다.
기존 U-Net의 가중치는 YOLO에 옮겨 쓸 수 없습니다. 공식 사전학습 가중치에서 새로 미세조정합니다.

## 파일 구조

```text
computer_vision/
├─ data/
│  ├─ raw/magnetic_tile_dataset/           # 원본 공유
│  └─ processed/
│     ├─ segmentation/splits/*.csv         # U-Net과 같은 분할
│     └─ yolo26_semantic/                  # prepare_data.py로 생성
│        ├─ dataset.yaml                  # 이미지/마스크 위치와 클래스
│        ├─ manifest.json                 # 크기, 개수, 원본 CSV 해시
│        ├─ images/{train,val,test}/*.png  # RGB로 복제한 흑백 영상
│        └─ masks/{train,val,test}/*.png   # 클래스 ID 0~5
├─ unet_defect_segmentation/
└─ yolo26_defect_segmentation/
   ├─ requirements.txt
   ├─ src/
   │  ├─ config.py          # 경로, 클래스, 학습 기본값
   │  ├─ prepare_data.py    # 기존 CSV → images/masks와 YAML
   │  ├─ train.py           # 사전학습 모델 미세조정
   │  ├─ inference.py       # 모델 로드/예측/색상 시각화 공통 함수
   │  ├─ metrics.py         # U-Net과 같은 IoU/Dice 계산 방식
   │  ├─ evaluate.py        # 테스트 전체 평가 + 비교 이미지
   │  └─ predict.py         # 정답 없이 한 장 추론
   ├─ tests/test_pipeline.py
   ├─ models/best.pt        # 가장 최근에 성공한 학습의 best 복사본
   ├─ runs/                # 각 학습 실행의 가중치, 로그, 설정
   └─ outputs/
      ├─ test/             # metrics.json, predictions/, visualizations/
      └─ single/           # predictions/, visualizations/
```

`model.py`, `dataset.py`, `losses.py`를 새로 구현할 필요는 없습니다.
Ultralytics가 모델, 데이터 로더, 손실 함수를 제공합니다.

## 실행 순서

아래 명령은 `computer_vision` 루트에서 실행합니다. PyTorch가 동작하는 환경을 사용하세요.

```powershell
python -m pip install -U -r yolo26_defect_segmentation/requirements.txt
python yolo26_defect_segmentation/src/prepare_data.py
python yolo26_defect_segmentation/src/train.py
python yolo26_defect_segmentation/src/evaluate.py
```

첫 학습 때 공식 가중치를 다운로드하므로 인터넷이 필요합니다.
기존 U-Net 체크포인트가 있어도 YOLO 학습은 별도로 실행해야 합니다.
기본값은 nano 모델, 256×256, batch 8, 최대 40 epoch, patience 8입니다.
`--device 0`은 첫 CUDA GPU, `--device cpu`는 CPU를 선택합니다.
생략하면 Ultralytics가 장치를 선택합니다. 메모리가 부족하면 `--batch 4`로 줄이세요.

```powershell
python yolo26_defect_segmentation/src/train.py --epochs 100 --batch 4 --device 0
```

학습별 원본 가중치는 `runs/<실험명>/weights/best.pt`에 보존됩니다.
`models/best.pt`와 `best_metadata.json`은 가장 최근 성공한 학습 결과로 갱신됩니다.
과거 실험을 평가하려면 `evaluate.py --weights <해당 best.pt 경로>`를 사용합니다.
변환 데이터가 이미 있으면 `prepare_data.py`는 덮어쓰지 않습니다. 기존 YAML로 학습하면 됩니다.

## 결과 보기

`outputs/test/visualizations/*.png`는 **원본 | 정답 | 예측** 순서입니다.
터미널과 `outputs/test/metrics.json`에 클래스별 IoU/Dice와 배경 제외 평균을 저장합니다.
`predictions/*.png`는 색상이 아니라 클래스 번호를 저장하므로 눈으로 보면 거의 검습니다.

정답 없이 이미지 한 장을 추론하려면:

```powershell
python yolo26_defect_segmentation/src/predict.py --source data/raw/magnetic_tile_dataset/MT_Blowhole/Imgs/exp2_num_40424.jpg
```

`outputs/single/visualizations/`에 **원본(크기 조정) | 예측**을 저장합니다.
이 스크립트는 학습과 동일하게 정사각형으로 resize하므로 출력 좌표는 원본 크기 좌표가 아닙니다.

## 데이터와 비교 조건

- 기존 CSV를 재분할하지 않습니다. 정상 영상은 배경 0 마스크를 생성합니다.
- 원본 마스크를 nearest 방식으로 resize한 뒤, `>127` 영역을 결함 종류별 ID로 바꿉니다.
- ID는 `Background=0, Blowhole=1, Break=2, Crack=3, Fray=4, Uneven=5`입니다.
- YOLO semantic에서 **255는 ignore**이므로 원본 0/255 마스크를 그대로 사용하면 안 됩니다.
- 사전학습 모델의 입력을 유지하도록 grayscale을 RGB 3채널에 복제합니다.
- U-Net처럼 기본 256×256 평가 격자에서 전체 test 픽셀을 누적합니다.
- 배경을 제외한 mIoU/Dice의 평균 방식도 U-Net과 같습니다. 정답과 예측 양쪽에 없는 클래스만 제외합니다.
- 사전학습, 내부 손실, 스케줄러와 모델 선택 기준 등은 다릅니다. 이 비교는 전체 학습 파이프라인 비교이며, 구조만의 우열을 분리하는 실험은 아닙니다.
- Ultralytics의 기본 검증/체크포인트 선택 점수와 이 프로젝트의 **foreground mIoU**를 혼동하지 마세요. 최종 비교에는 `evaluate.py` 출력을 사용합니다.
- 테스트 결과로 모델을 반복 선택하지 말고, 설정 선택에는 validation을 사용하세요.

512 해상도 실험은 원본에서 별도 데이터를 만들어 진행합니다. 256 이미지를 확대하지 않습니다.

```powershell
python yolo26_defect_segmentation/src/prepare_data.py --size 512 --output data/processed/yolo26_semantic_512
python yolo26_defect_segmentation/src/train.py --data data/processed/yolo26_semantic_512/dataset.yaml --name yolo26n_sem_512
python yolo26_defect_segmentation/src/evaluate.py --data data/processed/yolo26_semantic_512/dataset.yaml --output yolo26_defect_segmentation/outputs/test_512
```

512 평가는 마스크 해상도도 다르므로 기존 256 U-Net 점수와 엄밀히 같은 조건은 아닙니다.

## 검증

```powershell
python -m unittest discover -s yolo26_defect_segmentation/tests -v
```

테스트는 정상 마스크, 결함 ID 변환, 잘못된 라벨, 데이터 중복, 혼동행렬/지표,
semantic 결과 처리와 시각화 저장을 검증합니다. 가중치 다운로드나 학습은 수행하지 않습니다.

구현 시 Ultralytics 8.4.164에서 샘플 2장, 1 epoch의 실제 학습 → 체크포인트 로드 →
평가 → 단일 이미지 추론을 확인했습니다. 이 점검은 무작위 초기화 모델을 사용한 실행 검증으로,
전체 데이터의 사전학습 미세조정 성능을 의미하지 않습니다. 변환한 784장의 입력 영상과 마스크는
기존 U-Net Dataset의 평가용 출력과 픽셀 단위로 일치하는 것도 확인했습니다.

공식 API 참고:
- https://docs.ultralytics.com/tasks/semantic/
- https://docs.ultralytics.com/datasets/semantic/
