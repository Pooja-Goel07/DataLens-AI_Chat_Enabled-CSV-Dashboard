import { createContext, useContext, useState, type ReactNode } from 'react'
import { api, tokenStore } from './api'

interface AuthState {
  token: string | null
  user: string | null
  login: (username: string, password: string) => Promise<void>
  signup: (username: string, password: string) => Promise<void>
  logout: () => void
}

const AuthContext = createContext<AuthState | null>(null)

export function AuthProvider({ children }: { children: ReactNode }) {
  const [token, setToken] = useState<string | null>(tokenStore.get())
  const [user, setUser] = useState<string | null>(tokenStore.getUser())

  const login = async (username: string, password: string) => {
    const t = await api.login(username, password)
    tokenStore.set(t, username)
    setToken(t)
    setUser(username)
  }

  const signup = async (username: string, password: string) => {
    await api.signup(username, password)
  }

  const logout = () => {
    tokenStore.clear()
    setToken(null)
    setUser(null)
  }

  return (
    <AuthContext.Provider value={{ token, user, login, signup, logout }}>
      {children}
    </AuthContext.Provider>
  )
}

export function useAuth() {
  const ctx = useContext(AuthContext)
  if (!ctx) throw new Error('useAuth must be used within AuthProvider')
  return ctx
}
