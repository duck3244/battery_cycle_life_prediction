<script setup>
import { computed, ref } from 'vue'
import { Line } from 'vue-chartjs'
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Tooltip,
  Legend,
  Filler,
} from 'chart.js'
import { predictMat } from '../api.js'

ChartJS.register(CategoryScale, LinearScale, PointElement, LineElement, Tooltip, Legend, Filler)

defineProps({
  ready: { type: Boolean, default: false },
})

const file = ref(null)
const result = ref(null)
const error = ref(null)
const loading = ref(false)

function onFile(event) {
  file.value = event.target.files?.[0] ?? null
}

async function submit() {
  if (!file.value) return
  loading.value = true
  error.value = null
  result.value = null
  try {
    result.value = await predictMat(file.value)
  } catch (err) {
    error.value = err.response?.data?.detail || err.message
  } finally {
    loading.value = false
  }
}

const chartData = computed(() => {
  if (!result.value) return null
  return {
    labels: result.value.predictions.map((_, i) => i + 1),
    datasets: [
      {
        label: 'Predicted RUL (cycles)',
        data: result.value.predictions,
        borderColor: '#38bdf8',
        backgroundColor: 'rgba(56,189,248,0.2)',
        fill: true,
        tension: 0.15,
        pointRadius: 0,
      },
    ],
  }
})

const chartOptions = {
  responsive: true,
  maintainAspectRatio: false,
  scales: {
    x: { title: { display: true, text: 'Cycle index' }, ticks: { color: '#94a3b8' } },
    y: { title: { display: true, text: 'RUL (cycles)' }, ticks: { color: '#94a3b8' } },
  },
  plugins: { legend: { labels: { color: '#e2e8f0' } } },
}
</script>

<template>
  <section class="card">
    <h2>Predict from .mat</h2>
    <p class="muted">Uploads a MathWorks-format battery file and returns RUL predictions per cycle.</p>

    <div v-if="!ready" class="warning">
      Train the model first — the server needs a checkpoint and normalization params.
    </div>

    <form @submit.prevent="submit" class="form">
      <input type="file" accept=".mat" @change="onFile" :disabled="!ready" />
      <button type="submit" :disabled="!ready || !file || loading">
        {{ loading ? 'Predicting…' : 'Run prediction' }}
      </button>
    </form>

    <p v-if="error" class="pill err">{{ error }}</p>

    <div v-if="result" class="output">
      <div class="summary">
        <span><span class="muted">Count</span><strong>{{ result.count }}</strong></span>
        <span><span class="muted">Mean</span><strong>{{ result.mean.toFixed(1) }}</strong></span>
        <span><span class="muted">Min</span><strong>{{ result.min.toFixed(1) }}</strong></span>
        <span><span class="muted">Max</span><strong>{{ result.max.toFixed(1) }}</strong></span>
      </div>
      <div class="chart-box">
        <Line :data="chartData" :options="chartOptions" />
      </div>
    </div>
  </section>
</template>

<style scoped>
h2 { margin: 0 0 0.25rem; font-size: 1.15rem; }
.warning {
  margin-top: 0.75rem;
  padding: 0.6rem 0.8rem;
  border-radius: 0.4rem;
  border: 1px dashed var(--border);
  color: var(--muted);
  font-size: 0.9rem;
}
.form {
  margin-top: 1rem;
  display: flex;
  gap: 0.75rem;
  align-items: center;
}
.output {
  margin-top: 1rem;
  border-top: 1px solid var(--border);
  padding-top: 1rem;
  display: flex;
  flex-direction: column;
  gap: 0.75rem;
}
.summary {
  display: grid;
  grid-template-columns: repeat(4, 1fr);
  gap: 0.5rem 1rem;
}
.summary span {
  display: flex;
  flex-direction: column;
}
.chart-box {
  height: 280px;
}
</style>
