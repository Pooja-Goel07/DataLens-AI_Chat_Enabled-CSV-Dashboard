import { useState } from 'react'
import { api, API_BASE, UnauthorizedError } from '../api'
import { useToast } from '../toast'
import type { ReportData } from '../types'
import { Button, Card, Spinner } from './ui'

interface Props {
  activeTable: string | null
  activeTableName: string | null
  onUnauthorized: () => void
}

export function Report({ activeTable, activeTableName, onUnauthorized }: Props) {
  const toast = useToast()
  const [report, setReport] = useState<ReportData | null>(null)
  const [busy, setBusy] = useState(false)

  const generate = async () => {
    if (!activeTable) return
    setBusy(true)
    setReport(null)
    try {
      const data = await api.generateReport(activeTable)
      setReport(data)
    } catch (err) {
      if (err instanceof UnauthorizedError) return onUnauthorized()
      toast.error(err instanceof Error ? err.message : 'Report failed')
    } finally {
      setBusy(false)
    }
  }

  if (!activeTable) {
    return (
      <div className="flex h-full items-center justify-center text-sm text-neutral-400">
        Select or upload a dataset to generate a report.
      </div>
    )
  }

  return (
    <div className="h-full overflow-y-auto p-6">
      <div className="mb-5 flex items-center justify-between">
        <div>
          <h2 className="text-base font-semibold">{activeTableName}</h2>
          <p className="text-xs text-neutral-400">Automated analysis report</p>
        </div>
        <Button onClick={generate} disabled={busy}>
          {busy && <Spinner />}
          {report ? 'Regenerate' : 'Generate report'}
        </Button>
      </div>

      {busy && (
        <div className="flex items-center gap-2 text-sm text-neutral-400">
          <Spinner /> Analyzing dataset & building charts…
        </div>
      )}

      {report && (
        <div className="space-y-6">
          <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
            <Stat label="Rows" value={report.total_rows.toLocaleString()} />
            <Stat label="Columns" value={String(report.total_columns)} />
            <Stat label="Outliers" value={String(Object.keys(report.outliers || {}).length)} />
            <Stat label="File" value={report.file_name || '—'} />
          </div>

          {report.summary && (
            <Card className="p-5">
              <h3 className="mb-2 text-sm font-semibold">Executive summary</h3>
              <div className="whitespace-pre-wrap text-sm leading-relaxed text-neutral-700 dark:text-neutral-300">
                {report.summary}
              </div>
            </Card>
          )}

          <Card className="p-5">
            <h3 className="mb-3 text-sm font-semibold">Columns ({report.columns.length})</h3>
            <div className="flex flex-wrap gap-2">
              {report.columns.map((c) => (
                <span
                  key={c}
                  className="rounded-md bg-neutral-100 px-2 py-1 text-xs dark:bg-neutral-800"
                >
                  {c}
                </span>
              ))}
            </div>
          </Card>

          {report.graph_urls && report.graph_urls.length > 0 && (
            <div>
              <h3 className="mb-3 text-sm font-semibold">
                Visualizations ({report.graph_urls.length})
              </h3>
              <div className="grid grid-cols-1 gap-4 lg:grid-cols-2">
                {report.graph_urls.map((url, i) => (
                  <Card key={i} className="overflow-hidden p-2">
                    <img
                      src={`${API_BASE}${url}`}
                      alt={`chart ${i + 1}`}
                      loading="lazy"
                      className="w-full rounded-md"
                    />
                  </Card>
                ))}
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  )
}

function Stat({ label, value }: { label: string; value: string }) {
  return (
    <Card className="p-4">
      <div className="text-xs text-neutral-400">{label}</div>
      <div className="mt-1 truncate text-lg font-semibold">{value}</div>
    </Card>
  )
}
