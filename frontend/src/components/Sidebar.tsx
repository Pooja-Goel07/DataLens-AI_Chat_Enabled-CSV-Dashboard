import { useRef, useState } from 'react'
import { api } from '../api'
import { useAuth } from '../auth'
import { useToast } from '../toast'
import type { DocumentDetail, TableDetail } from '../types'
import { Button, Input, Modal, Spinner } from './ui'

interface Props {
  tables: TableDetail[]
  activeTable: string | null
  documents: DocumentDetail[]
  onSwitchTable: (t: string) => void
  onTablesChanged: () => void
  onDocumentsChanged: () => void
}

export function Sidebar({
  tables,
  activeTable,
  documents,
  onSwitchTable,
  onTablesChanged,
  onDocumentsChanged,
}: Props) {
  const { user, logout } = useAuth()
  const toast = useToast()
  const [csvOpen, setCsvOpen] = useState(false)
  const [docOpen, setDocOpen] = useState(false)

  return (
    <aside className="flex w-72 shrink-0 flex-col border-r border-neutral-200 bg-white dark:border-neutral-800 dark:bg-neutral-900">
      <div className="flex items-center gap-2 border-b border-neutral-200 px-4 py-3 dark:border-neutral-800">
        <span className="text-xl">📊</span>
        <span className="font-semibold">DataLens</span>
      </div>

      <div className="flex-1 overflow-y-auto p-3">
        {/* Datasets */}
        <div className="mb-2 flex items-center justify-between">
          <h3 className="text-xs font-semibold uppercase tracking-wide text-neutral-400">
            Datasets
          </h3>
          <Button size="sm" variant="ghost" onClick={() => setCsvOpen(true)}>
            + CSV
          </Button>
        </div>
        {tables.length === 0 && (
          <p className="px-1 py-2 text-xs text-neutral-400">No datasets yet. Upload a CSV.</p>
        )}
        <ul className="space-y-1">
          {tables.map((t) => (
            <li key={t.table_name}>
              <button
                onClick={() => onSwitchTable(t.table_name)}
                className={`group flex w-full items-center justify-between rounded-lg px-2.5 py-2 text-left text-sm transition-colors ${
                  t.table_name === activeTable
                    ? 'bg-indigo-500/10 text-indigo-700 dark:text-indigo-300'
                    : 'hover:bg-neutral-100 dark:hover:bg-neutral-800'
                }`}
              >
                <span className="min-w-0">
                  <span className="block truncate font-medium">{t.original_name}</span>
                  <span className="block text-[11px] text-neutral-400">
                    {t.rows} rows · {t.columns} cols
                  </span>
                </span>
                <span
                  role="button"
                  onClick={async (e) => {
                    e.stopPropagation()
                    if (!confirm(`Delete dataset "${t.original_name}"?`)) return
                    try {
                      await api.deleteTable(t.table_name)
                      toast.success('Dataset deleted')
                      onTablesChanged()
                    } catch (err) {
                      toast.error(err instanceof Error ? err.message : 'Delete failed')
                    }
                  }}
                  className="hidden text-neutral-400 hover:text-red-500 group-hover:block"
                >
                  ✕
                </span>
              </button>
            </li>
          ))}
        </ul>

        {/* Documents */}
        <div className="mb-2 mt-6 flex items-center justify-between">
          <h3 className="text-xs font-semibold uppercase tracking-wide text-neutral-400">
            Documents (RAG)
          </h3>
          <Button size="sm" variant="ghost" onClick={() => setDocOpen(true)}>
            + Doc
          </Button>
        </div>
        {documents.length === 0 && (
          <p className="px-1 py-2 text-xs text-neutral-400">
            No documents. Upload a PDF/TXT to ask about it.
          </p>
        )}
        <ul className="space-y-1">
          {documents.map((d) => (
            <li
              key={d.doc_name}
              className="group flex items-center justify-between rounded-lg px-2.5 py-2 text-sm hover:bg-neutral-100 dark:hover:bg-neutral-800"
            >
              <span className="min-w-0">
                <span className="block truncate font-medium">{d.doc_name}</span>
                <span className="block text-[11px] text-neutral-400">{d.chunks} chunks</span>
              </span>
              <button
                onClick={async () => {
                  if (!confirm(`Delete document "${d.doc_name}"?`)) return
                  try {
                    await api.deleteDocument(d.doc_name)
                    toast.success('Document deleted')
                    onDocumentsChanged()
                  } catch (err) {
                    toast.error(err instanceof Error ? err.message : 'Delete failed')
                  }
                }}
                className="hidden text-neutral-400 hover:text-red-500 group-hover:block"
              >
                ✕
              </button>
            </li>
          ))}
        </ul>
      </div>

      <div className="flex items-center justify-between border-t border-neutral-200 px-4 py-3 dark:border-neutral-800">
        <span className="truncate text-sm text-neutral-500">👤 {user}</span>
        <Button size="sm" variant="ghost" onClick={logout}>
          Log out
        </Button>
      </div>

      <UploadDialog
        open={csvOpen}
        onClose={() => setCsvOpen(false)}
        title="Upload CSV dataset"
        accept=".csv"
        nameLabel="Dataset name"
        onUpload={(file, name) => api.uploadCsv(file, name)}
        onDone={() => {
          setCsvOpen(false)
          onTablesChanged()
        }}
      />
      <UploadDialog
        open={docOpen}
        onClose={() => setDocOpen(false)}
        title="Upload document (PDF / TXT)"
        accept=".pdf,.txt"
        nameLabel="Document name"
        onUpload={(file, name) => api.uploadDocument(file, name)}
        onDone={() => {
          setDocOpen(false)
          onDocumentsChanged()
        }}
      />
    </aside>
  )
}

function UploadDialog({
  open,
  onClose,
  title,
  accept,
  nameLabel,
  onUpload,
  onDone,
}: {
  open: boolean
  onClose: () => void
  title: string
  accept: string
  nameLabel: string
  onUpload: (file: File, name: string) => Promise<unknown>
  onDone: () => void
}) {
  const toast = useToast()
  const [name, setName] = useState('')
  const [busy, setBusy] = useState(false)
  const fileRef = useRef<HTMLInputElement>(null)

  const submit = async (e: React.FormEvent) => {
    e.preventDefault()
    const file = fileRef.current?.files?.[0]
    if (!file || !name.trim()) {
      toast.error('Pick a file and enter a name')
      return
    }
    setBusy(true)
    try {
      await onUpload(file, name.trim())
      toast.success('Uploaded and processed')
      setName('')
      if (fileRef.current) fileRef.current.value = ''
      onDone()
    } catch (err) {
      toast.error(err instanceof Error ? err.message : 'Upload failed')
    } finally {
      setBusy(false)
    }
  }

  return (
    <Modal open={open} onClose={onClose} title={title}>
      <form onSubmit={submit} className="space-y-3">
        <div>
          <label className="mb-1 block text-xs font-medium text-neutral-500">{nameLabel}</label>
          <Input value={name} onChange={(e) => setName(e.target.value)} placeholder={nameLabel} />
        </div>
        <div>
          <label className="mb-1 block text-xs font-medium text-neutral-500">File</label>
          <input
            ref={fileRef}
            type="file"
            accept={accept}
            className="block w-full text-sm text-neutral-500 file:mr-3 file:rounded-md file:border-0 file:bg-indigo-600 file:px-3 file:py-1.5 file:text-sm file:font-medium file:text-white hover:file:bg-indigo-500"
          />
        </div>
        <Button type="submit" className="w-full" disabled={busy}>
          {busy && <Spinner />}
          Upload
        </Button>
      </form>
    </Modal>
  )
}
