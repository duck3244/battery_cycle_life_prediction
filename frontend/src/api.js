import axios from 'axios'

const client = axios.create({
  baseURL: '/api',
  timeout: 600_000, // training can take a while
})

export async function fetchHealth() {
  const { data } = await client.get('/health')
  return data
}

export async function predictMat(file) {
  const form = new FormData()
  form.append('file', file)
  const { data } = await client.post('/predict', form, {
    headers: { 'Content-Type': 'multipart/form-data' },
  })
  return data
}

export async function triggerTraining(payload) {
  const { data } = await client.post('/train', payload)
  return data
}

export async function fetchReport() {
  const { data } = await client.get('/results/report')
  return data
}

export async function fetchPlots() {
  const { data } = await client.get('/results/plots')
  return data.plots
}

export function plotUrl(name) {
  return `/api/results/plots/${encodeURIComponent(name)}`
}
