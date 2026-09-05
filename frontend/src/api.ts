// Typed API client for the DataLens FastAPI backend.
import type { AskResponse, DocumentDetail, ReportData, TableDetail } from './types'

export const API_BASE =
  (import.meta.env.VITE_API_BASE_URL as string | undefined)?.replace(/\/$/, '') ||
  'http://localhost:8000'

const TOKEN_KEY = 'datalens_token'
const USER_KEY = 'datalens_user'

export const tokenStore = {
  get: () => localStorage.getItem(TOKEN_KEY),
  getUser: () => localStorage.getItem(USER_KEY),
  set: (token: string, user: string) => {
    localStorage.setItem(TOKEN_KEY, token)
    localStorage.setItem(USER_KEY, user)
  },
  clear: () => {
    localStorage.removeItem(TOKEN_KEY)
    localStorage.removeItem(USER_KEY)
  },
}

/** Thrown when the backend responds 401 — the caller should log the user out. */
export class UnauthorizedError extends Error {}

async function request<T>(path: string, init: RequestInit = {}, auth = true): Promise<T> {
  const headers = new Headers(init.headers)
  if (auth) {
    const token = tokenStore.get()
    if (token) headers.set('Authorization', `Bearer ${token}`)
  }
  const res = await fetch(`${API_BASE}${path}`, { ...init, headers })
  if (res.status === 401) throw new UnauthorizedError('Session expired')
  const data = res.status === 204 ? null : await res.json().catch(() => null)
  if (!res.ok) {
    const detail = (data && (data.detail || data.error)) || `Request failed (${res.status})`
    throw new Error(typeof detail === 'string' ? detail : JSON.stringify(detail))
  }
  return data as T
}

export const api = {
  async login(username: string, password: string): Promise<string> {
    const body = new URLSearchParams({ grant_type: 'password', username, password })
    const data = await request<{ access_token: string }>(
      '/api/token',
      { method: 'POST', headers: { 'Content-Type': 'application/x-www-form-urlencoded' }, body },
      false,
    )
    return data.access_token
  },

  signup(username: string, password: string) {
    return request(
      '/api/signup/',
      {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ username, password }),
      },
      false,
    )
  },

  async listTables(): Promise<{ table_details: TableDetail[]; active_table: string | null }> {
    return request('/api/list_tables/')
  },

  switchTable(table_name: string) {
    return request('/api/switch_table/', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ table_name }),
    })
  },

  deleteTable(table_name: string) {
    return request(`/api/table/${encodeURIComponent(table_name)}`, { method: 'DELETE' })
  },

  uploadCsv(file: File, tableName: string) {
    const fd = new FormData()
    fd.append('file', file)
    fd.append('table_name', tableName)
    return request('/api/upload_csv/', { method: 'POST', body: fd })
  },

  ask(message: string, table_name: string | null): Promise<AskResponse> {
    return request('/api/ask', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ message, table_name }),
    })
  },

  generateReport(table_name: string): Promise<ReportData> {
    return request(`/api/generate_report/?table_name=${encodeURIComponent(table_name)}`, {
      method: 'POST',
    })
  },

  listDocuments(): Promise<{ documents: DocumentDetail[] }> {
    return request('/api/list_documents/')
  },

  uploadDocument(file: File, docName: string) {
    const fd = new FormData()
    fd.append('file', file)
    fd.append('doc_name', docName)
    return request('/api/upload_document/', { method: 'POST', body: fd })
  },

  deleteDocument(doc_name: string) {
    return request(`/api/document/${encodeURIComponent(doc_name)}`, { method: 'DELETE' })
  },
}
