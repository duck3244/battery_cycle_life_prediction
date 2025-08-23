# Battery Cycle Life Prediction Using Deep Learning

이 프로젝트는 MATLAB의 배터리 사이클 수명 예측 예제를 Python으로 변환한 것입니다. 딥러닝(CNN)을 사용하여 리튬이온 배터리의 잔여 사이클 수명을 예측합니다.

 - https://kr.mathworks.com/help/predmaint/ug/battery-cycle-life-prediction-using-deep-learning.html

## 📁 프로젝트 구조

```
battery_cycle_life_prediction/
├── config.py              # 설정 파일
├── data_loader.py         # 데이터 로딩 및 다운로드
├── data_preprocessor.py   # 데이터 전처리 및 특성 엔지니어링
├── model.py              # CNN 모델 아키텍처 및 훈련
├── evaluator.py          # 모델 평가 및 성능 지표
├── visualizer.py         # 데이터 시각화 및 분석
├── main.py               # 메인 실행 스크립트
├── requirements.txt      # 필요한 패키지 목록
└── README.md            # 프로젝트 문서
```

## 🚀 설치 및 실행

### 1. 환경 설정

```bash
# 가상환경 생성 (권장)
python -m venv battery_env
source battery_env/bin/activate  # Windows: battery_env\Scripts\activate

# 필요한 패키지 설치
pip install -r requirements.txt
```

### 2. 실행 방법

#### 전체 파이프라인 실행 (합성 데이터 사용)
```bash
python main.py --mode full
```

#### 실제 데이터 다운로드 및 사용
```bash
python main.py --mode full --use-real-data
```

#### 커스텀 설정으로 실행
```bash
python main.py --mode full --epochs 50 --batch-size 128 --no-plots
```

#### 모델 훈련만 실행
```bash
python main.py --mode train --epochs 100
```

## 📊 주요 기능

### 1. 데이터 처리 (`data_loader.py`, `data_preprocessor.py`)
- **실제 데이터 다운로드**: MathWorks에서 제공하는 1.2GB 배터리 데이터셋
- **합성 데이터 생성**: 테스트를 위한 합성 배터리 데이터 생성
- **방전 데이터 추출**: 3.6V~2.0V 방전 구간 데이터 추출 및 스무딩
- **선형 보간**: 900개 포인트로 균등 보간 및 30×30 행렬로 reshape
- **CNN 입력 형태**: 30×30×3 (전압, 온도, 방전용량) 이미지 형태 변환

### 2. 모델 아키텍처 (`model.py`)
- **CNN 구조**: 5개 Convolutional layer + Layer Normalization + ReLU
- **최적화**: Adam optimizer, MAE loss function
- **콜백**: Early stopping, Learning rate scheduling, Model checkpointing
- **특성맵 시각화**: 각 층의 feature map 추출 및 시각화 기능

### 3. 평가 및 시각화 (`evaluator.py`, `visualizer.py`)
- **성능 지표**: RMSE, MAE, MAPE, R² Score 등
- **예측 vs 실제값**: 산점도 및 완벽 예측 라인
- **잔차 분석**: 잔차 분포, Q-Q plot, 오차 분석
- **훈련 히스토리**: Loss 및 metrics 변화 추이
- **데이터 대시보드**: 종합적인 데이터 개요 및 통계

## 🔧 설정 사용자 정의

`config.py`에서 다음 설정들을 조정할 수 있습니다:

```python
# 모델 설정
MAX_BATTERY_LIFE = 2000    # 출력 정규화용 최대 배터리 수명
BATCH_SIZE = 256           # 배치 크기
EPOCHS = 100               # 최대 에포크
LEARNING_RATE = 0.001      # 학습률

# 데이터 설정
VOLTAGE_RANGE = (3.6, 2.0)  # 방전 전압 범위
INTERPOLATION_POINTS = 900   # 보간 포인트 수
RESHAPE_SIZE = 30           # 재구성 행렬 크기

# 아키텍처 설정
CONV_FILTERS = [8, 16, 32, 32, 32]  # 각 층의 필터 수
```

## 📈 예상 결과

합성 데이터에서의 일반적인 성능:
- **RMSE**: ~70-100 사이클
- **MAE**: ~50-80 사이클  
- **MAPE**: ~15-25%
- **R² Score**: ~0.7-0.9

실제 데이터에서는 원본 MATLAB 예제와 유사한 성능이 기대됩니다.

## 📁 출력 파일

실행 후 생성되는 파일들:

```
results/
├── dataset_overview.png         # 데이터셋 개요 대시보드
├── sample_measurements.png      # 샘플 배터리 측정 데이터
├── interpolated_data.png        # 보간된 데이터 시각화
├── training_history.png         # 훈련 과정 히스토리
├── predictions_vs_actual.png    # 예측 vs 실제값
├── residual_analysis.png        # 잔차 분석
├── error_distribution.png       # 오차 분포 분석
├── feature_maps.png            # CNN 특성맵 시각화
└── evaluation_report.json      # 종합 평가 보고서

models/
└── battery_model.h5            # 훈련된 모델
```

## 🔍 사용 예제

### Python 스크립트에서 직접 사용

```python
from main import BatteryCycleLifePipeline

# 파이프라인 초기화
pipeline = BatteryCycleLifePipeline()

# 전체 파이프라인 실행
success = pipeline.run_complete_pipeline(
    use_synthetic=True,    # 합성 데이터 사용
    download_real=False,   # 실제 데이터 다운로드 안함
    create_plots=True,     # 시각화 생성
    save_results=True,     # 결과 저장
    epochs=50,             # 50 에포크로 훈련
    batch_size=256         # 배치 크기 256
)
```

### 개별 구성 요소 사용

```python
from config import Config
from data_loader import DataLoader
from model import BatteryLifeModel

# 설정 및 초기화
config = Config()
loader = DataLoader(config)
model = BatteryLifeModel(config)

# 데이터 로드
discharge_data = loader.create_synthetic_data(num_batteries=20)

# 모델 생성 및 훈련
model.create_model()
# ... 데이터 전처리 후
# model.train(train_data, train_labels, val_data, val_labels)
```

## 🛠️ 커맨드 라인 옵션

```bash
# 사용 가능한 모든 옵션 보기
python main.py --help

# 주요 옵션들:
--mode {train,predict,full}    # 실행 모드 선택
--use-real-data               # 실제 데이터 다운로드 및 사용
--epochs EPOCHS               # 훈련 에포크 수
--batch-size BATCH_SIZE       # 배치 크기
--no-plots                    # 시각화 생성 건너뛰기
--no-save                     # 파일 저장 건너뛰기
```

## ⚠️ 주의사항

1. **메모리 요구사항**: 전체 데이터셋 로드 시 8GB+ RAM 권장
2. **실제 데이터 다운로드**: ~1.2GB 크기, 다운로드 시간 고려
3. **훈련 시간**: GPU 사용 권장 (CPU로도 가능하지만 시간이 오래 걸림)
4. **Python 버전**: Python 3.8+ 권장

## 📚 참고 자료

- [원본 MATLAB 예제](https://kr.mathworks.com/help/predmaint/ug/battery-cycle-life-prediction-using-deep-learning.html)
- [배터리 데이터셋 논문](https://doi.org/10.1038/s41560-019-0356-8)
- [TensorFlow 문서](https://www.tensorflow.org/)
