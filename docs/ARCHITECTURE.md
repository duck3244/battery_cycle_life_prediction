# Architecture

이 문서는 Battery Cycle Life Prediction 프로젝트의 아키텍처를 설명합니다.
클래스 단위 관계와 시퀀스는 [`UML.md`](./UML.md)를 참고하세요.

## 1. 아키텍처 개요

프로젝트는 세 레이어로 구성된 **모놀리식 ML + 얇은 웹 래퍼** 구조입니다.

```
┌──────────────────────────────────────────────────────────────┐
│                      Browser (Vue 3 SPA)                     │
│  HealthPanel · TrainPanel · PredictPanel · ResultsPanel      │
└──────────────────────────┬───────────────────────────────────┘
                           │  axios, JSON / multipart
                           │  (Vite dev 프록시: /api → :8000)
                           ▼
┌──────────────────────────────────────────────────────────────┐
│          FastAPI Layer (backend/api/server.py, :8000)        │
│  /api/health  /api/train  /api/predict  /api/results/*       │
│           ── Pydantic schemas · CORS · file upload ──        │
└──────────────────────────┬───────────────────────────────────┘
                           │  직접 함수 호출 (in-process)
                           ▼
┌──────────────────────────────────────────────────────────────┐
│        Pipeline Orchestrator (backend/main.py)               │
│              BatteryCycleLifePipeline                        │
└─────┬────────┬────────┬────────┬────────┬────────────────────┘
      │        │        │        │        │
      ▼        ▼        ▼        ▼        ▼
   Loader  Preproc   Model   Evaluator  Visualizer
            (+norm)  (CNN)
      ── 공용 유틸: config.py · utils.py ──
      ── 아티팩트: models/, results/, data/ ──
```

- **Layer 경계**: 프론트엔드는 REST만 통해 접근하므로 백엔드 내부 변경은
  API 계약이 유지되는 한 SPA에 영향을 주지 않습니다.
- **단일 프로세스**: FastAPI 서버가 파이프라인 객체를 프로세스 내에서
  싱글톤으로 보유합니다 (`get_pipeline()`). 별도 워커/큐 없음 — 학습은
  요청 스레드에서 동기 실행되며 axios timeout 10 분으로 커버합니다.
- **모델 상태**: 파일 시스템(`backend/models/`)이 런타임 간 상태 저장소.
  학습 후 `battery_model.h5` + `norm_params.npz`가 남고, 재기동 후에도
  `/api/predict`가 바로 동작합니다.

## 2. 디렉터리 레이아웃

```
battery_cycle_life_prediction/
├── backend/
│   ├── api/
│   │   ├── __init__.py
│   │   └── server.py           # FastAPI app + 엔드포인트
│   ├── run_server.py           # uvicorn 런처
│   ├── main.py                 # BatteryCycleLifePipeline + CLI
│   ├── config.py               # Config (정적 설정)
│   ├── utils.py                # 로거·시드·JSON sanitize
│   ├── data_loader.py          # DataLoader
│   ├── data_preprocessor.py    # DataPreprocessor
│   ├── model.py                # BatteryLifeModel (tf.keras)
│   ├── evaluator.py            # ModelEvaluator (메트릭 + 플롯)
│   ├── visualizer.py           # DataVisualizer (데이터 탐색 플롯)
│   ├── tests/                  # pytest (13개)
│   ├── data/   models/   results/   # ← 런타임 산출물 (gitignored)
│   └── requirements.txt
├── frontend/
│   ├── src/
│   │   ├── App.vue             # 최상위 레이아웃 + health 새로고침
│   │   ├── api.js              # axios 래퍼 (/api/*)
│   │   ├── main.js  style.css
│   │   └── components/
│   │       ├── HealthPanel.vue
│   │       ├── TrainPanel.vue
│   │       ├── PredictPanel.vue
│   │       └── ResultsPanel.vue
│   ├── vite.config.js          # :5173, /api → :8000 프록시
│   ├── package.json / index.html
│   └── README.md
├── docs/                       # ← 이 문서
├── README.md   DATA_LICENSE.md   LICENSE
```

## 3. 백엔드 레이어

### 3.1 역할별 컴포넌트

| 계층 | 컴포넌트 | 파일 | 책임 |
|---|---|---|---|
| 설정 / 공용 | `Config` | `config.py` | 정적 하이퍼파라미터·경로. 다른 모든 모듈이 의존 |
| 설정 / 공용 | `utils` | `utils.py` | `setup_logging`, `set_global_seed`, `sanitize_for_json` |
| 데이터 | `DataLoader` | `data_loader.py` | MathWorks ZIP 다운로드·`.mat` 로드·합성 데이터 생성·검증 |
| 데이터 | `DataPreprocessor` | `data_preprocessor.py` | 방전 구간 추출, 보간, 30×30×3 reshape, 배터리 단위 분할, 정규화 fit/apply/save/load |
| 모델 | `BatteryLifeModel` | `model.py` | Keras CNN 생성·학습·예측·평가·체크포인트 |
| 평가 | `ModelEvaluator` | `evaluator.py` | RMSE/MAE/MAPE/R²·잔차·예측-실제 플롯·JSON 리포트 |
| 시각화 | `DataVisualizer` | `visualizer.py` | 데이터셋 개요·전처리 단계·feature map 시각화 |
| 오케스트레이션 | `BatteryCycleLifePipeline` | `main.py` | 위 컴포넌트를 순차 실행. CLI + API 양쪽에서 재사용 |
| 웹 | `api.server` | `api/server.py` | FastAPI 엔드포인트·Pydantic 스키마·CORS·업로드 |

### 3.2 모듈 의존 방향

순환 없는 단일 방향 의존입니다.

```
server.py ─┐
           ▼
       main.py ──► data_loader · data_preprocessor · model · evaluator · visualizer
                        └── 모두 config · utils 에만 공통 의존
```

- `config` · `utils`는 말단(leaf) — 어떤 모듈에도 의존하지 않음.
- `main` 외에는 다른 컴포넌트를 import 하지 않음 → 개별 컴포넌트는
  단위 테스트로 격리 가능 (`tests/test_preprocessor.py`, `test_evaluator.py`,
  `test_utils.py`).

### 3.3 파이프라인 데이터 흐름

`BatteryCycleLifePipeline.run_complete_pipeline()`은 다음 순서로 실행합니다.

1. **load_data** — 합성 or `.mat` 다운로드 → `self.discharge_data`
2. **preprocess_data**
   1. `extract_discharge_data`: V/T/Qd → 방전 구간(3.6→2.0 V)
   2. `linear_interpolation`: 전압축 900 포인트 선형 보간 → 30×30 행렬
   3. `split_data_indices`: 배터리 인덱스를 train/val/test 로 disjoint 분할
   4. `reshape_for_cnn`: 3 채널 스택 → (N, 30, 30, 3) + RUL 라벨
   5. `normalize_data` (train 에서만 fit) → params → `save_norm_params` → `apply_normalization`을 val/test 에도 적용
3. **train_model** — `BatteryLifeModel.create_model` → `train` (EarlyStopping, ReduceLROnPlateau, ModelCheckpoint 콜백)
4. **evaluate_model** — `predict(test, rescale=True)` → `calculate_metrics`
5. **create_visualizations** — 데이터 개요·훈련 히스토리·예측 vs 실제·잔차·오차 분포·feature map PNG
6. **generate_report** — `create_evaluation_report` → `evaluation_report.json`

`predict_from_mat(mat_path)`은 3·4 단계에서만 재활용: `.mat` → 전처리 → `load_norm_params` 로 저장된 파라미터 적용 → `load_model` → `predict(rescale=True)`.

### 3.4 정규화 전략

누수(leakage)를 막기 위해:

- **Fit 시점**: `preprocess_data()`에서 **train split에만** fit.
- **저장 위치**: `backend/models/norm_params.npz` (`method`, `min`/`max`
  또는 `mean`/`std`).
- **Apply 시점**: val/test, 추론(`predict_from_mat`) 모두 `apply_normalization`
  으로 동일 파라미터 재사용.
- **교체 가능**: `Config.NORMALIZATION_METHOD = 'minmax' | 'zscore'`.

### 3.5 FastAPI 엔드포인트

| Method | Path | Request | Response | 핸들러 |
|---|---|---|---|---|
| GET | `/api/health` | — | `HealthResponse {status, model_available, norm_params_available, model_path, norm_path}` | `health()` |
| POST | `/api/train` | `TrainRequest {use_synthetic, epochs?, batch_size?, create_plots}` | `TrainResponse {success, metrics?}` | `train()` |
| POST | `/api/predict` | multipart `file: UploadFile (.mat)` | `PredictResponse {count, predictions, min, max, mean}` | `predict()` |
| GET | `/api/results/report` | — | 최신 `evaluation_report.json` (JSON) | `get_report()` |
| GET | `/api/results/plots` | — | `{plots: [filename, ...]}` | `list_plots()` |
| GET | `/api/results/plots/{name}` | — | PNG (FileResponse, path-traversal 차단) | `get_plot()` |

- **CORS**: Vite dev origin(`http://localhost:5173`, `http://127.0.0.1:5173`)만 허용.
- **파일 업로드**: `.mat` 확장자 검사 → `tempfile.NamedTemporaryFile`에 저장 →
  pipeline 호출 → finally에서 삭제.
- **싱글톤**: `_pipeline` 전역 + `get_pipeline()` 지연 초기화.

## 4. 프론트엔드 레이어

### 4.1 컴포넌트 트리

```
App.vue
├── HealthPanel.vue     (props: health, error; emits: refresh)
├── TrainPanel.vue      (emits: trained)
├── PredictPanel.vue    (props: ready)
└── ResultsPanel.vue    (props: modelAvailable)
```

- `App.vue`가 `/api/health`를 주기적으로 폴링하지 않고, 마운트 시 한 번 + 학습 완료/패널 요청 시 갱신. `health.model_available`·`norm_params_available`를 자식에게 전달해 Predict/Results 패널의 활성화 조건으로 사용.
- 각 패널은 자기 영역의 API 호출만 담당 — App은 오케스트레이션만 수행.

### 4.2 API 클라이언트 (`api.js`)

```js
axios.create({ baseURL: '/api', timeout: 600_000 })
```

| 함수 | 호출 | 용도 |
|---|---|---|
| `fetchHealth()` | GET `/api/health` | HealthPanel 새로고침 |
| `triggerTraining(payload)` | POST `/api/train` | TrainPanel 학습 실행 |
| `predictMat(file)` | POST `/api/predict` (multipart) | PredictPanel 업로드 |
| `fetchReport()` | GET `/api/results/report` | ResultsPanel metrics |
| `fetchPlots()` | GET `/api/results/plots` | ResultsPanel 플롯 리스트 |
| `plotUrl(name)` | URL 생성 | `<img :src>` 바인딩 |

Vite dev 서버(`:5173`)는 `/api`를 `http://localhost:8000`으로 프록시하므로
CORS 이슈 없이 동일 origin 처럼 동작합니다. 프로덕션에서는 빌드된
`frontend/dist/`를 임의 정적 호스팅에 두고 `/api`만 백엔드로 역방향
프록시하면 됩니다.

## 5. 배포·실행 모드

| 모드 | 실행 | 엔트리 |
|---|---|---|
| CLI 학습/추론 | `python backend/main.py --mode full` | `backend.main:main` |
| REST 서버 | `python backend/run_server.py` | `api.server:app` (uvicorn) |
| 프론트엔드 개발 | `cd frontend && npm run dev` | Vite dev server :5173 |
| 프론트엔드 배포 | `npm run build` → `frontend/dist/` | 정적 호스팅 |
| 파이프라인 임포트 | `from main import BatteryCycleLifePipeline` | 스크립트 / 노트북 |

## 6. 지속 상태와 설정

- **런타임 산출물** (모두 gitignored):
  - `backend/data/` — 다운로드한 ZIP/`.mat`
  - `backend/models/battery_model.h5`, `battery_model_history.json`, `norm_params.npz`
  - `backend/results/*.png`, `evaluation_report.json`, 선택적 `predictions.csv`
- **설정**: `backend/config.py` (코드 상수). 환경 변수 사용 없음 — 파라미터
  변경은 `Config` 속성 수정 또는 CLI 플래그·`TrainRequest` JSON으로 오버라이드.
- **로깅**: `utils.setup_logging('INFO')` → 표준 `battery` 로거. 파일
  로그 경로는 `log_file` 인자로 주입 가능.

## 7. 설계상의 주요 결정

| 결정 | 근거 |
|---|---|
| 싱글톤 파이프라인 | Keras 모델 로드 비용(초 단위)을 요청마다 피하려면 프로세스 내 캐시가 필수. 싱글 유저/단일 인스턴스 가정이면 충분. |
| 배터리 단위 split | 사이클 단위로 섞으면 같은 배터리의 학습·평가 cycle이 양쪽에 들어가 리크. 배터리 인덱스 기반 분할로 원본 MATLAB 예제와 일치. |
| train-only normalization fit | val/test/추론에 train 통계 재사용 — scikit-learn `fit/transform` 규약과 동일. |
| 동기 학습 엔드포인트 | 학습은 초~분 단위. 별도 job queue 없이 axios timeout 10 분으로 커버. 장기화 시 Celery/RQ 도입 여지 있음. |
| Vue 3 Composition API | 컴포넌트 간 상태 최소화 (App → 자식 단방향 props). 글로벌 store 불필요. |
| `.h5`·`.npz` 파일시스템 저장 | 단일 인스턴스 가정. 수평 확장이 필요해지면 S3/MinIO + object store 경로로 교체 가능. |

## 8. 테스트 전략

`backend/tests/` 아래 13 개 pytest 케이스.

- `test_preprocessor.py` — split disjoint·중복 시 raise·minmax round-trip·
  save/load·scipy `.mat` 레이아웃 관용성 (parametrized).
- `test_evaluator.py` — 완전 예측의 R²=1, zero-truth MAPE 안전, JSON
  직렬화.
- `test_utils.py` — `sanitize_for_json`·`set_global_seed` 결정성.

FastAPI 레벨 테스트는 미구현 — 현재는 `curl` 스모크 테스트로 커버
(`/api/health`, `/api/train`, `/api/predict`, `/api/results/*`, 경로 탈출·
잘못된 확장자 등 에러 경로 포함).

## 9. 향후 확장 포인트

- `Config`를 `pydantic-settings` 기반으로 전환하여 환경 변수 오버라이드 지원.
- 학습 작업을 background task(Celery·RQ·BackgroundTasks)로 분리하고 상태
  조회 엔드포인트(`/api/train/{job_id}`) 추가.
- 모델 레지스트리(`models/*/version/`)와 `/api/predict?model=v2` 쿼리로
  A/B 지원.
- 모델·normalization 산출물을 object store로 옮기고 서버는 stateless화.
- Vue 쪽에 WebSocket 기반 학습 진행 스트리밍.
