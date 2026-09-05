import { useCallback, useEffect, useState } from 'react'
import { api, UnauthorizedError } from './api'
import { useAuth } from './auth'
import { useToast } from './toast'
import { Chat } from './components/Chat'
import { Login } from './components/Login'
import { Report } from './components/Report'
import { Sidebar } from './components/Sidebar'
import type { DocumentDetail, TableDetail } from './types'

export default function App() {
  const { token, logout } = useAuth()
  if (!token) return <Login />
  return <Workspace onUnauthorized={logout} />
}

function Workspace({ onUnauthorized }: { onUnauthorized: () => void }) {
  const toast = useToast()
  const [tables, setTables] = useState<TableDetail[]>([])
  const [documents, setDocuments] = useState<DocumentDetail[]>([])
  const [activeTable, setActiveTable] = useState<string | null>(null)
  const [tab, setTab] = useState<'chat' | 'report'>('chat')

  const handleUnauthorized = useCallback(() => {
    toast.error('Session expired — please log in again.')
    onUnauthorized()
  }, [toast, onUnauthorized])

  const loadTables = useCallback(async () => {
    try {
      const data = await api.listTables()
      setTables(data.table_details || [])
      setActiveTable((cur) => cur ?? data.active_table ?? data.table_details?.[0]?.table_name ?? null)
    } catch (err) {
      if (err instanceof UnauthorizedError) handleUnauthorized()
    }
  }, [handleUnauthorized])

  const loadDocuments = useCallback(async () => {
    try {
      const data = await api.listDocuments()
      setDocuments(data.documents || [])
    } catch (err) {
      if (err instanceof UnauthorizedError) handleUnauthorized()
    }
  }, [handleUnauthorized])

  useEffect(() => {
    loadTables()
    loadDocuments()
  }, [loadTables, loadDocuments])

  const switchTable = async (t: string) => {
    setActiveTable(t)
    try {
      await api.switchTable(t)
    } catch (err) {
      if (err instanceof UnauthorizedError) return handleUnauthorized()
      toast.error(err instanceof Error ? err.message : 'Switch failed')
    }
  }

  const activeTableName =
    tables.find((t) => t.table_name === activeTable)?.original_name ?? activeTable

  return (
    <div className="flex h-full">
      <Sidebar
        tables={tables}
        activeTable={activeTable}
        documents={documents}
        onSwitchTable={switchTable}
        onTablesChanged={loadTables}
        onDocumentsChanged={loadDocuments}
      />

      <main className="flex min-w-0 flex-1 flex-col">
        <div className="flex items-center gap-1 border-b border-neutral-200 px-4 py-2 dark:border-neutral-800">
          {(['chat', 'report'] as const).map((t) => (
            <button
              key={t}
              onClick={() => setTab(t)}
              className={`rounded-lg px-3 py-1.5 text-sm font-medium capitalize transition-colors ${
                tab === t
                  ? 'bg-neutral-100 text-neutral-900 dark:bg-neutral-800 dark:text-white'
                  : 'text-neutral-500 hover:text-neutral-800 dark:hover:text-neutral-200'
              }`}
            >
              {t}
            </button>
          ))}
        </div>

        <div className="min-h-0 flex-1">
          {tab === 'chat' ? (
            <Chat
              activeTable={activeTable}
              activeTableName={activeTableName}
              onUnauthorized={handleUnauthorized}
            />
          ) : (
            <Report
              activeTable={activeTable}
              activeTableName={activeTableName}
              onUnauthorized={handleUnauthorized}
            />
          )}
        </div>
      </main>
    </div>
  )
}
