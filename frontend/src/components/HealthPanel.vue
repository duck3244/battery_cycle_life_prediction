<script setup>
defineProps({
  health: { type: Object, default: null },
  error: { type: String, default: null },
})
defineEmits(['refresh'])
</script>

<template>
  <section class="card">
    <div class="row">
      <h2>Backend status</h2>
      <button class="refresh" @click="$emit('refresh')">Refresh</button>
    </div>

    <p v-if="error" class="pill err">Unreachable — {{ error }}</p>

    <ul v-else-if="health" class="status-list">
      <li>
        <span class="muted">Service</span>
        <span class="pill ok">{{ health.status }}</span>
      </li>
      <li>
        <span class="muted">Model</span>
        <span :class="['pill', health.model_available ? 'ok' : 'err']">
          {{ health.model_available ? 'ready' : 'not trained' }}
        </span>
        <code>{{ health.model_path }}</code>
      </li>
      <li>
        <span class="muted">Norm params</span>
        <span :class="['pill', health.norm_params_available ? 'ok' : 'err']">
          {{ health.norm_params_available ? 'ready' : 'missing' }}
        </span>
        <code>{{ health.norm_path }}</code>
      </li>
    </ul>

    <p v-else class="muted">Checking…</p>
  </section>
</template>

<style scoped>
.row {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 0.75rem;
}
h2 { margin: 0; font-size: 1.15rem; }
.refresh { padding: 0.35rem 0.8rem; font-size: 0.85rem; }
.status-list {
  margin: 0;
  padding: 0;
  list-style: none;
  display: flex;
  flex-direction: column;
  gap: 0.5rem;
}
.status-list li {
  display: flex;
  gap: 0.75rem;
  align-items: center;
  font-size: 0.95rem;
}
code {
  font-size: 0.8rem;
  color: #94a3b8;
  font-family: ui-monospace, SFMono-Regular, monospace;
}
</style>
