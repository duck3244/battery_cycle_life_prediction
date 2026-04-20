<script setup>
import { onMounted, ref } from 'vue'
import { fetchHealth } from './api.js'
import HealthPanel from './components/HealthPanel.vue'
import TrainPanel from './components/TrainPanel.vue'
import PredictPanel from './components/PredictPanel.vue'
import ResultsPanel from './components/ResultsPanel.vue'

const health = ref(null)
const healthError = ref(null)

async function refreshHealth() {
  try {
    health.value = await fetchHealth()
    healthError.value = null
  } catch (err) {
    healthError.value = err.message || 'Backend unreachable'
    health.value = null
  }
}

onMounted(refreshHealth)
</script>

<template>
  <main class="app">
    <header>
      <h1>Battery Cycle Life Prediction</h1>
      <p class="muted">CNN-based Remaining Useful Life (RUL) estimator.</p>
    </header>

    <HealthPanel
      :health="health"
      :error="healthError"
      @refresh="refreshHealth"
    />

    <div class="grid">
      <TrainPanel @trained="refreshHealth" />
      <PredictPanel :ready="health?.model_available && health?.norm_params_available" />
    </div>

    <ResultsPanel :model-available="!!health?.model_available" />
  </main>
</template>

<style scoped>
.app {
  max-width: 1100px;
  margin: 0 auto;
  padding: 2rem 1.5rem 4rem;
  display: flex;
  flex-direction: column;
  gap: 1.25rem;
}

header h1 {
  margin: 0 0 0.25rem;
  font-size: 1.75rem;
}

.grid {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 1.25rem;
}

@media (max-width: 720px) {
  .grid {
    grid-template-columns: 1fr;
  }
}
</style>
