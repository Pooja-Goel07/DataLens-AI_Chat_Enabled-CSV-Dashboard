import { useState } from 'react'
import { useAuth } from '../auth'
import { useToast } from '../toast'
import { Button, Card, Input, Spinner } from './ui'

export function Login() {
  const { login, signup } = useAuth()
  const toast = useToast()
  const [mode, setMode] = useState<'login' | 'signup'>('login')
  const [username, setUsername] = useState('')
  const [password, setPassword] = useState('')
  const [busy, setBusy] = useState(false)

  const submit = async (e: React.FormEvent) => {
    e.preventDefault()
    if (!username || !password) return
    setBusy(true)
    try {
      if (mode === 'login') {
        await login(username, password)
        toast.success('Welcome back!')
      } else {
        await signup(username, password)
        toast.success('Account created — please log in.')
        setMode('login')
        setPassword('')
      }
    } catch (err) {
      toast.error(err instanceof Error ? err.message : 'Something went wrong')
    } finally {
      setBusy(false)
    }
  }

  return (
    <div className="flex min-h-full items-center justify-center p-4">
      <Card className="w-full max-w-sm p-6">
        <div className="mb-6 text-center">
          <div className="mx-auto mb-3 flex h-12 w-12 items-center justify-center rounded-xl bg-indigo-600 text-2xl">
            📊
          </div>
          <h1 className="text-lg font-semibold">DataLens</h1>
          <p className="text-sm text-neutral-500">Chat with your data & documents</p>
        </div>

        <div className="mb-4 grid grid-cols-2 gap-1 rounded-lg bg-neutral-100 p-1 dark:bg-neutral-800">
          {(['login', 'signup'] as const).map((m) => (
            <button
              key={m}
              onClick={() => setMode(m)}
              className={`rounded-md py-1.5 text-sm font-medium capitalize transition-colors ${
                mode === m
                  ? 'bg-white text-neutral-900 shadow-sm dark:bg-neutral-950 dark:text-white'
                  : 'text-neutral-500'
              }`}
            >
              {m === 'login' ? 'Log in' : 'Sign up'}
            </button>
          ))}
        </div>

        <form onSubmit={submit} className="space-y-3">
          <Input
            placeholder="Username"
            value={username}
            onChange={(e) => setUsername(e.target.value)}
            autoFocus
          />
          <Input
            type="password"
            placeholder="Password"
            value={password}
            onChange={(e) => setPassword(e.target.value)}
          />
          <Button type="submit" className="w-full" disabled={busy}>
            {busy && <Spinner />}
            {mode === 'login' ? 'Log in' : 'Create account'}
          </Button>
        </form>
      </Card>
    </div>
  )
}
