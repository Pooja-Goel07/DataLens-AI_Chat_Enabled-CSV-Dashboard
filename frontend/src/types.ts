// Shared response types (kept loose where the backend is dynamic).

export interface TableDetail {
  table_name: string
  original_name: string
  rows: number
  columns: number
  upload_time?: string | null
}

export interface DocumentDetail {
  doc_name: string
  file_name: string
  chunks: number
  upload_time?: string | null
}

// A single chat answer from POST /api/ask. The backend returns different shapes
// depending on the agent, so most fields are optional.
export interface AskResponse {
  agent?: string
  answer?: string
  sql_query?: string
  results?: unknown
  analysis?: string
  grounded?: boolean
  sources?: string[]
  active_dataset?: string
  blocked?: boolean
  error?: string
}

export interface ChatMessage {
  id: string
  role: 'user' | 'assistant'
  text: string
  meta?: AskResponse
}

export interface ReportData {
  table_name: string
  original_name?: string
  file_name?: string
  summary?: string
  outliers?: Record<string, any>
  graph_urls?: string[]
  total_rows: number
  total_columns: number
  columns: string[]
  generated_at: string
}
