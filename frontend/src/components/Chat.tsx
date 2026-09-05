import { useEffect, useRef, useState } from 'react'
import { api, UnauthorizedError } from '../api'
import { useToast } from '../toast'
import type { AskResponse, ChatMessage } from '../types'
import { AgentBadge, Button, Input, Spinner } from './ui'

interface Props {
  activeTable: string | null
  activeTableName: string | null
  onUnauthorized: () => void
}

function responseToText(r: AskResponse): string {
  if (r.answer) return r.answer
  if (r.analysis) return r.analysis
  if (r.error) return `⚠️ ${r.error}`
  if (r.results !== undefined) return 'Here are the results.'
  return '(no answer)'
}

let msgId = 1

export function Chat({ activeTable, activeTableName, onUnauthorized }: Props) {
  const toast = useToast()
  const [messages, setMessages] = useState<ChatMessage[]>([])
  const [input, setInput] = useState('')
  const [busy, setBusy] = useState(false)
  const endRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages, busy])

  const send = async () => {
    const text = input.trim()
    if (!text || busy) return
    setInput('')
    setMessages((m) => [...m, { id: String(msgId++), role: 'user', text }])
    setBusy(true)
    try {
      const resp = await api.ask(text, activeTable)
      setMessages((m) => [
        ...m,
        { id: String(msgId++), role: 'assistant', text: responseToText(resp), meta: resp },
      ])
    } catch (err) {
      if (err instanceof UnauthorizedError) {
        onUnauthorized()
        return
      }
      const message = err instanceof Error ? err.message : 'Request failed'
      toast.error(message)
      setMessages((m) => [
        ...m,
        { id: String(msgId++), role: 'assistant', text: `⚠️ ${message}` },
      ])
    } finally {
      setBusy(false)
    }
  }

  return (
    <div className="flex h-full flex-col">
      <div className="border-b border-neutral-200 px-5 py-3 dark:border-neutral-800">
        <h2 className="text-sm font-semibold">AI Assistant</h2>
        <p className="text-xs text-neutral-400">
          {activeTableName ? `Active dataset: ${activeTableName}` : 'No dataset selected'} · ask about
          data or your documents
        </p>
      </div>

      <div className="flex-1 space-y-4 overflow-y-auto px-5 py-4">
        {messages.length === 0 && (
          <div className="mx-auto mt-10 max-w-md text-center text-sm text-neutral-400">
            <div className="mb-2 text-3xl">🤖</div>
            Ask me things like <em>“what is the average price?”</em>, <em>“show top 5 rows”</em>, or
            a question about an uploaded document.
          </div>
        )}
        {messages.map((m) => (
          <MessageBubble key={m.id} message={m} />
        ))}
        {busy && (
          <div className="flex items-center gap-2 text-sm text-neutral-400">
            <Spinner /> thinking…
          </div>
        )}
        <div ref={endRef} />
      </div>

      <div className="border-t border-neutral-200 p-3 dark:border-neutral-800">
        <div className="flex gap-2">
          <Input
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={(e) => e.key === 'Enter' && send()}
            placeholder="Ask a question…"
            disabled={busy}
          />
          <Button onClick={send} disabled={busy || !input.trim()}>
            Send
          </Button>
        </div>
      </div>
    </div>
  )
}

function MessageBubble({ message }: { message: ChatMessage }) {
  const isUser = message.role === 'user'
  const meta = message.meta
  return (
    <div className={`flex ${isUser ? 'justify-end' : 'justify-start'}`}>
      <div className={`max-w-[80%] ${isUser ? 'order-2' : ''}`}>
        {!isUser && meta?.agent && (
          <div className="mb-1 flex items-center gap-2">
            <AgentBadge agent={meta.agent} />
            {meta.grounded === false && (
              <span className="text-[11px] text-amber-500">⚠ unverified</span>
            )}
            {meta.grounded === true && (
              <span className="text-[11px] text-emerald-500">✓ grounded</span>
            )}
          </div>
        )}
        <div
          className={`whitespace-pre-wrap rounded-2xl px-4 py-2.5 text-sm ${
            isUser
              ? 'bg-indigo-600 text-white'
              : 'bg-neutral-100 text-neutral-900 dark:bg-neutral-800 dark:text-neutral-100'
          }`}
        >
          {message.text}
        </div>

        {!isUser && meta?.sources && meta.sources.length > 0 && (
          <div className="mt-1.5 text-[11px] text-neutral-400">
            📄 Sources: {meta.sources.join(', ')}
          </div>
        )}

        {!isUser && meta?.sql_query && <SqlPanel meta={meta} />}
      </div>
    </div>
  )
}

function SqlPanel({ meta }: { meta: AskResponse }) {
  const [open, setOpen] = useState(false)
  return (
    <div className="mt-1.5">
      <button
        onClick={() => setOpen((o) => !o)}
        className="text-[11px] text-neutral-400 hover:text-neutral-600 dark:hover:text-neutral-200"
      >
        {open ? '▾' : '▸'} how I got this
      </button>
      {open && (
        <div className="mt-1 space-y-2 rounded-lg border border-neutral-200 bg-neutral-50 p-3 dark:border-neutral-800 dark:bg-neutral-950">
          <div>
            <div className="mb-1 text-[10px] font-semibold uppercase text-neutral-400">
              SQL executed
            </div>
            <pre className="overflow-x-auto whitespace-pre-wrap text-[11px] text-neutral-700 dark:text-neutral-300">
              {meta.sql_query}
            </pre>
          </div>
          {meta.results !== undefined && (
            <div>
              <div className="mb-1 text-[10px] font-semibold uppercase text-neutral-400">
                Raw results
              </div>
              <pre className="max-h-40 overflow-auto whitespace-pre-wrap text-[11px] text-neutral-600 dark:text-neutral-400">
                {typeof meta.results === 'string'
                  ? meta.results
                  : JSON.stringify(meta.results, null, 2)}
              </pre>
            </div>
          )}
        </div>
      )}
    </div>
  )
}
