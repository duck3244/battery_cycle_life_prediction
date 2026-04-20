<script setup>
import { reactive, ref } from 'vue'
import { triggerTraining } from '../api.js'

const emit = defineEmits(['trained'])

const form = reactive({
  use_synthetic: true,
  epochs: 10,
  batch_size: 64,
  create_plots: true,
})
const loading = ref(false)
const result = ref(null)
const error = ref(null)

async function submit() {
  loading.value = true
  error.value = null
  result.value = null
  try {
    result.value = await triggerTraining({ ...form })
    emit('trained')
  } catch (err) {
    error.value = err.response?.data?.detail || err.message
  } finally {
    loading.value = false
  }
}
</script>

<template>
  <section class="card">
    <h2>Train model</h2>
    <p class="muted">Runs the full pipeline: load → preprocess → train → evaluate.</p>

    <form @submit.prevent="submit" class="form">
      <label class="check">
        <input type="checkbox" v-model="form.use_synthetic" />
        Use synthetic data (skip MathWorks download)
      </label>

      <label>
        Epochs
        <input type="number" min="1" max="500" v-model.number="form.epochs" />
      </label>

      <label>
        Batch size
        <input type="number" min="1" max="1024" v-model.number="form.batch_size" />
      </label>

      <label class="check">
        <input type="checkbox" v-model="form.create_plots" />
        Generate diagnostic plots
      </label>

      <button type="submit" :disabled="loading">
        {{ loading ? 'Training…' : 'Start training' }}
      </button>
    </form>

    <p v-if="error" class="pill err">{{ error }}</p>

    <div v-if="result?.metrics" class="metrics">
      <h3>Test metrics</h3>
      <ul>
        <li><span class="muted">RMSE</span><strong>{{ result.metrics.RMSE?.toFixed(2) }}</strong></li>
        <li><span class="muted">MAE</span><strong>{{ result.metrics.MAE?.toFixed(2) }}</strong></li>
        <li><span class="muted">MAPE</span><strong>{{ result.metrics.MAPE?.toFixed(2) }}%</strong></li>
        <li><span class="muted">R²</span><strong>{{ result.metrics.R2_Score?.toFixed(4) }}</strong></li>
      </ul>
    </div>
  </section>
</template>

<style scoped>
h2 { margin: 0 0 0.25rem; font-size: 1.15rem; }
.form {
  display: flex;
  flex-direction: column;
  gap: 0.75rem;
  margin-top: 1rem;
}
label {
  display: flex;
  justify-content: space-between;
  align-items: center;
  gap: 1rem;
  font-size: 0.95rem;
}
label.check {
  justify-content: flex-start;
  gap: 0.5rem;
}
input[type='number'] {
  width: 8rem;
  text-align: right;
}
.metrics {
  margin-top: 1rem;
  border-top: 1px solid var(--border);
  padding-top: 1rem;
}
.metrics h3 { margin: 0 0 0.5rem; font-size: 1rem; }
.metrics ul {
  list-style: none;
  margin: 0;
  padding: 0;
  display: grid;
  grid-template-columns: repeat(2, 1fr);
  gap: 0.5rem 1rem;
}
.metrics li {
  display: flex;
  justify-content: space-between;
}
</style>
