# Battery Cycle Life Prediction

리튬이온 배터리의 잔여 사이클 수명(RUL)을 예측하는 CNN 파이프라인과 이를 감싸는
FastAPI + Vue 3 웹 애플리케이션입니다. MathWorks의 MATLAB 예제
*"Battery Cycle Life Prediction Using Deep Learning"*을 Python으로 포팅한 뒤,
학습/추론을 브라우저에서 조작할 수 있도록 REST API와 SPA를 추가했습니다.

원본 MATLAB 예제: <https://kr.mathworks.com/help/predmaint/ug/battery-cycle-life-prediction-using-deep-learning.html>

## 프로젝트 구조

```
battery_cycle_life_prediction/
├── backend/                    # Python ML 파이프라인 + FastAPI
│   ├── api/server.py           # REST 엔드포인트
│   ├── run_server.py           # uvicorn 런처 (:8000)
│   ├── main.py                 # CLI: train / predict / full
│   ├── config.py               # 하이퍼파라미터 및 경로
│   ├── data_loader.py          # .mat 다운로드·로드, 합성 데이터 생성
│   ├── data_preprocessor.py    # 방전 구간 추출·보간·정규화
│   ├── model.py                # CNN 모델 (5 conv + LayerNorm)
│   ├── evaluator.py            # RMSE / MAE / MAPE / R² 리포트
│   ├── visualizer.py           # 훈련/예측/잔차 플롯
│   ├── utils.py                # 로깅·시드·JSON sanitize
│   ├── tests/                  # pytest 스위트
│   └── requirements.txt
├── frontend/                   # Vue 3 + Vite SPA
│   ├── src/App.vue             # 레이아웃
│   ├── src/api.js              # axios 클라이언트 (/api/*)
│   └── src/components/
│       ├── HealthPanel.vue     # 백엔드 준비 상태
│       ├── TrainPanel.vue      # 학습 트리거
│       ├── PredictPanel.vue    # .mat 업로드 → 예측 차트
│       └── ResultsPanel.vue    # 최신 metrics + 저장된 플롯
├── DATA_LICENSE.md             # 데이터셋 attribution (CC BY 4.0)
└── README.md
```

## 빠른 시작

### 1. 백엔드 (FastAPI, http://localhost:8000)

```bash
cd backend

# conda 사용 시 (권장)
conda create -n py310_tf python=3.10
conda activate py310_tf
pip install -r requirements.txt

# 또는 venv
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python run_server.py
```

### 2. 프론트엔드 (Vue 3 + Vite, http://localhost:5173)

```bash
cd frontend
npm install
npm run dev          # Vite가 /api 요청을 :8000으로 프록시
```

두 서버가 모두 실행 중이면 브라우저에서 <http://localhost:5173>에 접속해
Health / Train / Predict / Results 패널을 사용할 수 있습니다.

## REST API

| Method | Path | 설명 |
|---|---|---|
| GET | `/api/health` | 모델·정규화 파라미터 적재 상태 |
| POST | `/api/train` | 학습 실행 `{use_synthetic, epochs, batch_size, create_plots}` |
| POST | `/api/predict` | `.mat` 업로드 → cycle별 RUL 예측 배열 |
| GET | `/api/results/report` | 최신 평가 리포트 (JSON) |
| GET | `/api/results/plots` | 저장된 플롯 파일 목록 |
| GET | `/api/results/plots/{name}` | 개별 PNG 다운로드 (path-traversal 차단) |

## CLI 사용

백엔드 디렉터리에서 실행합니다 (`cd backend`).

```bash
# 전체 파이프라인 (합성 데이터, 기본)
python main.py --mode full

# 실제 MathWorks 데이터셋 다운로드 후 학습 (~1.2 GB)
python main.py --mode full --use-real-data

# 학습만
python main.py --mode train --epochs 100

# .mat 파일로 추론
python main.py --mode predict --input-mat /path/to/battery.mat
```

주요 플래그: `--epochs`, `--batch-size`, `--no-plots`, `--no-save`.
`python main.py --help`로 전체 목록을 확인할 수 있습니다.

## 파이프라인 세부사항

- **방전 구간 추출**: 전압 3.6 V → 2.0 V 구간을 길이 3 uniform filter로 스무딩.
- **보간 & 리셰이프**: 전압축으로 900 포인트 선형 보간 → 30×30×3 텐서 (V/T/Qd).
- **분할**: `config.py`에서 train/val/test를 배터리 인덱스로 비겹치게 분리 (기본 step 8, start 0/1).
- **정규화**: train split에서만 fit, MinMax(기본) 또는 z-score. 파라미터는
  `backend/models/norm_params.npz`에 저장되어 추론 시 재사용.
- **모델**: 5-layer CNN + LayerNorm + ReLU, Adam + MAE loss, Early Stopping, LR scheduling.

`config.py`에서 다음 상수를 조정할 수 있습니다:

```python
MAX_BATTERY_LIFE = 2000
BATCH_SIZE = 256
EPOCHS = 100
LEARNING_RATE = 0.001
VOLTAGE_RANGE = (3.6, 2.0)
INTERPOLATION_POINTS = 900
RESHAPE_SIZE = 30
CONV_FILTERS = [8, 16, 32, 32, 32]
NORMALIZATION_METHOD = 'minmax'   # or 'zscore'
```

## 출력 파일

```
backend/
├── models/
│   ├── battery_model.h5         # 학습된 가중치
│   ├── battery_model_history.json
│   └── norm_params.npz          # train split에서 fit된 정규화 파라미터
└── results/
    ├── dataset_overview.png
    ├── sample_measurements.png
    ├── interpolated_data.png
    ├── training_history.png
    ├── predictions_vs_actual.png
    ├── residual_analysis.png
    ├── error_distribution.png
    ├── feature_maps.png
    └── evaluation_report.json
```

## 테스트

```bash
cd backend
pytest -q
```

현재 13개 테스트 (preprocessor split/normalize/extract, evaluator metrics,
utils seed·sanitize) 통과.

## 파이썬에서 직접 사용

```python
from main import BatteryCycleLifePipeline

pipeline = BatteryCycleLifePipeline()
pipeline.run_complete_pipeline(
    use_synthetic=True,
    create_plots=True,
    save_results=True,
    epochs=50,
    batch_size=256,
)

# 학습된 모델로 .mat 추론
preds = pipeline.predict_from_mat('/path/to/battery.mat', save_csv=True)
```

## 참고 성능

합성 데이터 기준 일반적인 범위:

| Metric | 값 |
|---|---|
| RMSE | 70–100 cycles |
| MAE | 50–80 cycles |
| MAPE | 15–25 % |
| R² | 0.7–0.9 |

실제 MathWorks 데이터셋에서는 원본 MATLAB 예제와 유사한 수치가 기대됩니다.

## 주의사항

- **메모리**: 실제 데이터셋 로드 시 8 GB+ RAM 권장.
- **다운로드**: 실제 데이터 ZIP은 약 1.2 GB.
- **GPU**: CPU로도 학습 가능하나 GPU가 현저히 빠릅니다.
- **Python**: 3.8 이상 (개발/검증은 3.10).

## 데이터셋 Attribution & License

이 프로젝트가 `--use-real-data` 모드에서 내려받는 데이터는
**Severson et al. (2019)** 의 고속 충전 배터리 데이터셋을 MathWorks가 전처리한
서브셋입니다.

- **Source**: [data.matr.io/1/projects/5c48dd2bc625d700019f3204](https://data.matr.io/1/projects/5c48dd2bc625d700019f3204)
- **License**: [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) — 상업적 사용 포함 허용, **attribution 필수**
- **Citation**:
  > Severson, K.A., Attia, P.M., Jin, N., Perkins, N., Jiang, B., Yang, Z.,
  > Chen, M.H., Aykol, M., Herring, P.K., Fraggedakis, D., Bazant, M.Z.,
  > Harris, S.J., Chueh, W.C., Braatz, R.D. *Data-driven prediction of
  > battery cycle life before capacity degradation*. **Nature Energy**
  > 4, 383–391 (2019). https://doi.org/10.1038/s41560-019-0356-8

파이프라인이 수행하는 변형과 재배포 조건은 [`DATA_LICENSE.md`](DATA_LICENSE.md)를 참고하세요.

## 참고 자료

- [원본 MATLAB 예제](https://kr.mathworks.com/help/predmaint/ug/battery-cycle-life-prediction-using-deep-learning.html)
- [Severson et al. 2019, Nature Energy](https://doi.org/10.1038/s41560-019-0356-8)
- [TensorFlow](https://www.tensorflow.org/) · [FastAPI](https://fastapi.tiangolo.com/) · [Vue 3](https://vuejs.org/)
