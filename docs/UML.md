# UML Diagrams

현재 구현의 구조(class)와 행위(sequence, state, component)를 Mermaid로 정리한
다이어그램입니다. GitHub·VS Code·JetBrains IDE의 Markdown 프리뷰에서 바로
렌더링됩니다. 시스템 개요와 설계 근거는 [`ARCHITECTURE.md`](./ARCHITECTURE.md)를
참고하세요.

## 목차

- [1. Component Diagram](#1-component-diagram) — 배포 단위 수준 관계
- [2. Class Diagram (Backend)](#2-class-diagram-backend) — 파이썬 클래스·메서드
- [3. Class Diagram (API schemas)](#3-class-diagram-api-schemas) — Pydantic 모델
- [4. Class Diagram (Frontend components)](#4-class-diagram-frontend-components) — Vue 컴포넌트
- [5. Sequence: Training Flow](#5-sequence-training-flow)
- [6. Sequence: Prediction Flow](#6-sequence-prediction-flow)
- [7. State Diagram: Pipeline Artifacts](#7-state-diagram-pipeline-artifacts)
- [8. Data Flow (Preprocessing)](#8-data-flow-preprocessing)

---

## 1. Component Diagram

배포 단위(브라우저 SPA, FastAPI 프로세스, 파일 시스템)와 외부 의존성을
표시합니다.

```mermaid
flowchart LR
    subgraph Browser["Browser (SPA)"]
        APP[App.vue]
        HP[HealthPanel]
        TP[TrainPanel]
        PP[PredictPanel]
        RP[ResultsPanel]
        AX[api.js<br/>axios client]
        APP --> HP & TP & PP & RP
        TP --> AX
        PP --> AX
        RP --> AX
        HP --> AX
    end

    subgraph Vite["Vite dev server :5173"]
        PROXY["proxy /api/* → :8000"]
    end

    subgraph Server["FastAPI :8000 (uvicorn)"]
        API["api/server.py"]
        PL[BatteryCycleLifePipeline]
        API --> PL
        PL --> L[DataLoader]
        PL --> P[DataPreprocessor]
        PL --> M[BatteryLifeModel]
        PL --> E[ModelEvaluator]
        PL --> V[DataVisualizer]
    end

    subgraph FS["File System (backend/)"]
        D[(data/*.mat)]
        MD[(models/*.h5<br/>norm_params.npz)]
        R[(results/*.png<br/>evaluation_report.json)]
    end

    subgraph Ext["External"]
        MW["MathWorks CDN<br/>batteryDischargeData.zip"]
    end

    AX --> PROXY --> API
    L -->|download| MW
    L -->|read/write| D
    M -->|load/save| MD
    P -->|load/save norm| MD
    E -->|write| R
    V -->|write| R
    API -->|stream| R
```

---

## 2. Class Diagram (Backend)

주요 Python 클래스와 메서드, 집계 관계입니다. 화살표는 `has-a`(composition),
점선은 `uses`(의존) 관계입니다.

```mermaid
classDiagram
    class Config {
        <<static>>
        +DATA_URL: str
        +DATA_DIR: str
        +DATA_FILE: str
        +MAX_BATTERY_LIFE: int
        +VOLTAGE_RANGE: tuple
        +INTERPOLATION_POINTS: int
        +RESHAPE_SIZE: int
        +NUM_CHANNELS: int
        +BATCH_SIZE: int
        +EPOCHS: int
        +LEARNING_RATE: float
        +NORMALIZATION_METHOD: str
        +NORM_PARAMS_PATH: str
        +CONV_FILTERS: list
        +MODEL_SAVE_PATH: str
        +RESULTS_DIR: str
        +create_directories()$ void
    }

    class DataLoader {
        -config: Config
        +__init__(config)
        +download_data(url, force_download) bool
        +load_battery_data(data_path) ndarray
        +create_synthetic_data(num_batteries) list
        +get_battery_info(discharge_data) dict
        +validate_data(discharge_data) bool
    }

    class DataPreprocessor {
        -config: Config
        +__init__(config)
        +extract_discharge_data(battery_data) list
        +linear_interpolation(discharge_data) tuple
        +reshape_for_cnn(V, T, Qd, battery_indices) tuple
        +split_data_indices(num_batteries) tuple
        +normalize_data(data, method) tuple
        +apply_normalization(data, params) ndarray
        +denormalize_data(data, params) ndarray
        +save_norm_params(params, path) str
        +load_norm_params(path) dict
        -_iter_struct(arr) list
        -_unwrap_field(record, field) any
        -_extract_array(data) ndarray
        -_extract_discharge_portion(V, T, Qd) tuple
        -_interpolate_cycle_data(volt, temp, qd, volt_range) tuple
    }

    class BatteryLifeModel {
        -config: Config
        -model: keras.Model
        -history: History
        -is_trained: bool
        +__init__(config)
        +create_model(input_shape) Model
        +get_model_summary() str
        +create_callbacks(val_data) list
        +train(train_data, train_labels, val_data, val_labels, epochs, batch_size) dict
        +predict(data, rescale) ndarray
        +evaluate(test_data, test_labels) dict
        +save_model(filepath) bool
        +load_model(filepath) bool
        +get_layer_output(data, layer_name) ndarray
        +get_feature_maps(data, conv_layer_names) dict
    }

    class ModelEvaluator {
        -config: Config
        +__init__(config)
        +calculate_metrics(y_true, y_pred) dict
        +print_metrics(metrics, title) void
        +plot_predictions_vs_actual(y_true, y_pred, title, save_path) Figure
        +plot_residuals(y_true, y_pred, title, save_path) Figure
        +plot_error_distribution(y_true, y_pred, title, save_path) Figure
        +plot_training_history(history, title, save_path) Figure
        +create_evaluation_report(y_true, y_pred, history, model_info, save_path) dict
        +compare_models(results_dict, save_path) DataFrame
    }

    class DataVisualizer {
        -config: Config
        +__init__(config)
        +plot_battery_measurements(...) Figure
        +plot_interpolated_data(...) Figure
        +plot_voltage_temperature_relationship(...) Figure
        +plot_cycle_life_distribution(...) Figure
        +plot_data_statistics(data_info, title, save_path) Figure
        +plot_feature_maps(feature_maps, sample_idx, max_filters, title, save_path) Figure
        +plot_data_preprocessing_pipeline(...) Figure
        +create_summary_dashboard(discharge_data, data_info, title, save_path) Figure
    }

    class BatteryCycleLifePipeline {
        -config: Config
        -data_loader: DataLoader
        -preprocessor: DataPreprocessor
        -model: BatteryLifeModel
        -evaluator: ModelEvaluator
        -visualizer: DataVisualizer
        -discharge_data: list
        -train_data / val_data / test_data: ndarray
        -train_labels / val_labels / test_labels: ndarray
        +__init__(config_path)
        +load_data(use_synthetic, download_real) bool
        +preprocess_data() bool
        +train_model(epochs, batch_size) bool
        +evaluate_model() bool
        +create_visualizations(save_plots) bool
        +generate_report(save_report) bool
        +run_complete_pipeline(...) bool
        +load_and_predict(model_path, data) ndarray
        +predict_from_mat(mat_path, model_path, norm_path, save_csv) ndarray
    }

    class utils {
        <<module>>
        +setup_logging(level, log_file) Logger
        +set_global_seed(seed) void
        +sanitize_for_json(value) any
    }

    BatteryCycleLifePipeline *-- DataLoader
    BatteryCycleLifePipeline *-- DataPreprocessor
    BatteryCycleLifePipeline *-- BatteryLifeModel
    BatteryCycleLifePipeline *-- ModelEvaluator
    BatteryCycleLifePipeline *-- DataVisualizer

    DataLoader ..> Config
    DataPreprocessor ..> Config
    BatteryLifeModel ..> Config
    ModelEvaluator ..> Config
    DataVisualizer ..> Config
    BatteryCycleLifePipeline ..> Config

    DataLoader ..> utils
    DataPreprocessor ..> utils
    BatteryLifeModel ..> utils
    ModelEvaluator ..> utils
    BatteryCycleLifePipeline ..> utils
```

---

## 3. Class Diagram (API schemas)

FastAPI 레이어의 Pydantic 모델과 엔드포인트입니다.

```mermaid
classDiagram
    class HealthResponse {
        +status: str
        +model_available: bool
        +norm_params_available: bool
        +model_path: str
        +norm_path: str
    }

    class TrainRequest {
        +use_synthetic: bool = true
        +epochs: int?
        +batch_size: int?
        +create_plots: bool = true
    }

    class TrainResponse {
        +success: bool
        +metrics: dict?
    }

    class PredictResponse {
        +count: int
        +predictions: list~float~
        +min: float
        +max: float
        +mean: float
    }

    class ApiServer {
        <<FastAPI>>
        -_pipeline: BatteryCycleLifePipeline
        +get_pipeline() BatteryCycleLifePipeline
        +health() HealthResponse
        +train(req: TrainRequest) TrainResponse
        +predict(file: UploadFile) PredictResponse
        +get_report() FileResponse
        +list_plots() dict
        +get_plot(name: str) FileResponse
    }

    ApiServer ..> HealthResponse : returns
    ApiServer ..> TrainRequest : accepts
    ApiServer ..> TrainResponse : returns
    ApiServer ..> PredictResponse : returns
    ApiServer ..> BatteryCycleLifePipeline : singleton
```

---

## 4. Class Diagram (Frontend components)

Vue 컴포넌트 계층과 props·events·API 호출입니다.

```mermaid
classDiagram
    class App {
        <<component>>
        -health: Ref~HealthResponse?~
        -healthError: Ref~string?~
        +refreshHealth() Promise
        +onMounted() void
    }

    class HealthPanel {
        <<component>>
        +props_health
        +props_error
        +emit_refresh()
    }

    class TrainPanel {
        <<component>>
        -form_use_synthetic: bool
        -form_epochs: int
        -form_batch_size: int
        -form_create_plots: bool
        -loading: bool
        -result: TrainResponse
        -error: string
        +submit() Promise
        +emit_trained()
    }

    class PredictPanel {
        <<component>>
        +props_ready: bool
        -file: File
        -result: PredictResponse
        -loading: bool
        -error: string
        -chartData
        -chartOptions
        +onFile(event) void
        +submit() Promise
    }

    class ResultsPanel {
        <<component>>
        +props_modelAvailable: bool
        -plots: string[]
        -report: object
        -error: string
        +load() Promise
    }

    class ApiClient {
        <<module>>
        -baseURL: string
        -timeout: int
        +fetchHealth() Promise
        +triggerTraining(payload) Promise
        +predictMat(file) Promise
        +fetchReport() Promise
        +fetchPlots() Promise
        +plotUrl(name) string
    }

    App *-- HealthPanel
    App *-- TrainPanel
    App *-- PredictPanel
    App *-- ResultsPanel

    HealthPanel ..> ApiClient : (via parent refresh)
    TrainPanel ..> ApiClient : triggerTraining
    PredictPanel ..> ApiClient : predictMat, plotUrl
    ResultsPanel ..> ApiClient : fetchReport, fetchPlots, plotUrl
    App ..> ApiClient : fetchHealth
```

---

## 5. Sequence: Training Flow

사용자가 TrainPanel에서 `Train` 버튼을 눌렀을 때의 전체 흐름.

```mermaid
sequenceDiagram
    actor User
    participant TP as TrainPanel.vue
    participant AX as api.js (axios)
    participant SRV as FastAPI /api/train
    participant PL as BatteryCycleLifePipeline
    participant L as DataLoader
    participant P as DataPreprocessor
    participant M as BatteryLifeModel
    participant E as ModelEvaluator
    participant FS as backend/models & results/

    User->>TP: 폼 제출 (use_synthetic, epochs, batch_size)
    TP->>AX: triggerTraining(payload)
    AX->>SRV: POST /api/train (JSON)
    SRV->>PL: get_pipeline()
    SRV->>PL: load_data(use_synthetic)
    PL->>L: create_synthetic_data() or load_battery_data()
    L-->>PL: discharge_data

    SRV->>PL: preprocess_data()
    PL->>P: extract_discharge_data
    PL->>P: linear_interpolation
    PL->>P: split_data_indices
    PL->>P: reshape_for_cnn (train/val/test)
    PL->>P: normalize_data(train)   # fit
    P->>FS: save_norm_params → norm_params.npz
    PL->>P: apply_normalization(val, test)

    SRV->>PL: train_model(epochs, batch_size)
    PL->>M: create_model()
    PL->>M: train(...)
    M->>FS: ModelCheckpoint → battery_model.h5

    SRV->>PL: evaluate_model()
    PL->>M: predict(test, rescale=true)
    PL->>E: calculate_metrics(y_true, y_pred)

    alt create_plots=true
        SRV->>PL: create_visualizations()
        PL->>E: plot_predictions_vs_actual, plot_residuals, ...
        E->>FS: *.png
        SRV->>PL: generate_report()
        PL->>E: create_evaluation_report
        E->>FS: evaluation_report.json
    end

    SRV-->>AX: 200 { success, metrics }
    AX-->>TP: TrainResponse
    TP->>TP: emit('trained')
    TP-->>User: 메트릭 표시 (RMSE/MAE/MAPE/R²)
```

---

## 6. Sequence: Prediction Flow

사용자가 PredictPanel에서 `.mat` 파일을 업로드했을 때.

```mermaid
sequenceDiagram
    actor User
    participant PP as PredictPanel.vue
    participant AX as api.js
    participant SRV as FastAPI /api/predict
    participant PL as BatteryCycleLifePipeline
    participant L as DataLoader
    participant P as DataPreprocessor
    participant M as BatteryLifeModel
    participant FS as backend/models

    User->>PP: 파일 선택 + Submit
    PP->>AX: predictMat(file)
    AX->>SRV: POST /api/predict (multipart)
    SRV->>SRV: validate .mat 확장자
    SRV->>SRV: NamedTemporaryFile(suffix=.mat) 저장
    SRV->>PL: get_pipeline()
    SRV->>PL: predict_from_mat(tmp_path, save_csv=false)

    PL->>L: load_battery_data(tmp_path)
    L-->>PL: raw batteryDischargeData

    PL->>P: extract_discharge_data(raw)
    PL->>P: linear_interpolation
    PL->>P: reshape_for_cnn(all indices)
    alt no usable cycles
        PL-->>SRV: RuntimeError "No usable cycles"
        SRV-->>AX: 422 detail
    else ok
        PL->>P: load_norm_params
        P->>FS: read norm_params.npz
        PL->>P: apply_normalization(data, params)
        PL->>M: load_model
        M->>FS: read battery_model.h5
        PL->>M: predict(data, rescale=true)
        M-->>PL: predictions (cycles)
        PL-->>SRV: ndarray
    end

    SRV->>SRV: predictions.tolist(), min/max/mean
    SRV-->>AX: 200 PredictResponse
    AX-->>PP: { count, predictions, min, max, mean }
    PP->>PP: Chart.js 라인차트 렌더링
    PP-->>User: 사이클별 RUL 차트
```

---

## 7. State Diagram: Pipeline Artifacts

파일 시스템의 상태 변화에 따라 API가 허용하는 동작이 달라집니다. `/api/health`가
반환하는 `model_available` · `norm_params_available` 조합으로 현재 상태를
판단합니다.

```mermaid
stateDiagram-v2
    [*] --> Empty

    Empty: No model, no norm
    Partial: model.h5 only<br/>(legacy or manual)
    Ready: model.h5 + norm_params.npz
    Trained: Ready + results/*.png + report.json

    Empty --> Ready: POST /api/train
    Partial --> Ready: POST /api/train<br/>(refit normalization)
    Ready --> Trained: create_plots=true
    Trained --> Ready: delete results/
    Ready --> Ready: POST /api/predict

    note right of Empty
        /api/predict → 500
        "Failed to load model"
    end note

    note right of Partial
        /api/predict → 500
        norm_params 누락
    end note

    note right of Ready
        /api/predict 가능
        /api/results/* 는 일부만 (report 없음)
    end note

    note right of Trained
        모든 엔드포인트 정상
    end note
```

---

## 8. Data Flow (Preprocessing)

원시 `.mat`/합성 데이터가 CNN 입력으로 변환되는 파이프라인입니다. Mermaid
flowchart로 shape 변화도 함께 표기합니다.

```mermaid
flowchart TD
    A["raw battery data<br/>list of dict V/T/Qd<br/>cycles of variable length"] --> B["extract_discharge_data<br/>3.6V to 2.0V 구간 + uniform_filter1d"]
    B --> C["linear_interpolation<br/>전압축 900포인트<br/>→ 30x30 행렬"]
    C --> D["reshape_for_cnn<br/>stack V/T/Qd<br/>→ N x 30 x 30 x 3"]
    D --> E["split_data_indices<br/>배터리 단위 train/val/test"]
    E --> F["normalize_data<br/>train에만 fit"]
    F --> G["save_norm_params<br/>→ norm_params.npz"]
    F --> H["apply_normalization<br/>val/test"]
    D --> I["RUL labels<br/>y / MAX_BATTERY_LIFE"]

    H --> J["CNN 입력: N x 30 x 30 x 3"]
    I --> J
    G -. read at inference .-> K["predict_from_mat"]
    J --> K
```
