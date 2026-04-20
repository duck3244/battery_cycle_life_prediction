# Battery Cycle Life Prediction — Frontend

Vue 3 + Vite SPA that talks to the FastAPI backend.

## Setup

```bash
cd frontend
npm install
npm run dev         # http://localhost:5173 (proxies /api -> :8000)
```

Make sure the backend is running in parallel:

```bash
cd backend
pip install -r requirements.txt
python run_server.py
```

## Build

```bash
npm run build       # outputs to dist/
npm run preview     # serve the production build locally
```

## Layout

- `src/App.vue` — top-level layout
- `src/api.js` — axios client targeting `/api/*`
- `src/components/HealthPanel.vue` — backend status
- `src/components/TrainPanel.vue` — trigger a training run
- `src/components/PredictPanel.vue` — upload .mat and chart predictions
- `src/components/ResultsPanel.vue` — latest metrics + saved plots
