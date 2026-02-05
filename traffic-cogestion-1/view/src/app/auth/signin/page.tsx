"use client"

import { useState } from "react"
import Link from "next/link"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Eye, EyeOff, LogIn, AlertCircle } from "lucide-react"

export default function SignInPage() {
  const [email, setEmail] = useState("demo@example.com")
  const [password, setPassword] = useState("demo")
  const [showPassword, setShowPassword] = useState(false)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const handleSignIn = async (e: React.FormEvent) => {
    e.preventDefault()

    if (!email || !password) {
      setError("Please fill in all fields")
      return
    }

    setLoading(true)
    setError(null)

    // Demo mode: Simulate network delay
    setTimeout(() => {
      // Mock validation
      if (email === 'demo@example.com' && password === 'demo') {
        // Success
        const user = {
          email: 'demo@example.com',
          user_metadata: { full_name: 'Demo User' },
          role: 'user'
        }
        localStorage.setItem('traffic_user', JSON.stringify(user))
        window.location.href = '/dashboard' // Force reload to update nav state
      } else if (email === 'admin@traffic.com' && password === 'admin') {
        // Admin Success
        const user = {
          email: 'admin@traffic.com',
          user_metadata: { full_name: 'Admin User' },
          role: 'admin'
        }
        localStorage.setItem('traffic_user', JSON.stringify(user))
        window.location.href = '/dashboard'
      } else {
        setError("Invalid credentials. Please use demo credentials.")
        setLoading(false)
      }
    }, 1000)
  }

  return (
    <>
      <div className="min-h-screen bg-background flex items-center justify-center p-6 relative z-10">
        <div className="absolute inset-0 bg-slate-950/50 -z-10"></div>
        <Card className="w-full max-w-md border-slate-700 bg-slate-900/80 backdrop-blur relative z-10 shadow-xl">
          <CardHeader className="text-center">
            <CardTitle className="text-2xl font-bold text-white">Sign In</CardTitle>
            <CardDescription className="text-slate-400">
              Access your traffic prediction dashboard
            </CardDescription>
          </CardHeader>
          <CardContent>
            <form onSubmit={handleSignIn} className="space-y-4" noValidate>
              {error && (
                <div className="flex items-center gap-2 p-3 rounded-lg bg-red-500/10 border border-red-500/30 text-red-400">
                  <AlertCircle className="w-4 h-4" />
                  <span className="text-sm">{error}</span>
                </div>
              )}

              <div className="bg-blue-500/10 border border-blue-500/20 rounded-lg p-3 text-sm text-blue-300 mb-4">
                <p className="font-semibold mb-1">Demo Credentials:</p>
                <p>Email: <span className="font-mono text-white">demo@example.com</span></p>
                <p>Password: <span className="font-mono text-white">demo</span></p>
              </div>

              <div className="space-y-2">
                <Label htmlFor="email" className="text-slate-300">Email</Label>
                <Input
                  id="email"
                  type="email"
                  placeholder="Enter your email"
                  value={email}
                  onChange={(e) => setEmail(e.target.value)}
                  autoComplete="email"
                  className="bg-slate-800 border-slate-600 text-white placeholder-slate-400 focus:border-blue-500 focus:ring-blue-500/20"
                />
              </div>

              <div className="space-y-2">
                <Label htmlFor="password" className="text-slate-300">Password</Label>
                <div className="relative">
                  <Input
                    id="password"
                    type={showPassword ? "text" : "password"}
                    placeholder="Enter your password"
                    value={password}
                    onChange={(e) => setPassword(e.target.value)}
                    autoComplete="current-password"
                    className="bg-slate-800 border-slate-600 text-white placeholder-slate-400 pr-10 focus:border-blue-500 focus:ring-blue-500/20"
                  />
                  <button
                    type="button"
                    onClick={() => setShowPassword(!showPassword)}
                    className="absolute right-3 top-1/2 -translate-y-1/2 text-slate-400 hover:text-slate-300"
                  >
                    {showPassword ? <EyeOff className="w-4 h-4" /> : <Eye className="w-4 h-4" />}
                  </button>
                </div>
              </div>

              <Button
                type="submit"
                disabled={loading}
                className="w-full bg-blue-600 hover:bg-blue-700 text-white"
              >
                {loading ? (
                  <div className="flex items-center gap-2">
                    <div className="w-4 h-4 border-2 border-white/30 border-t-white rounded-full animate-spin" />
                    Signing in...
                  </div>
                ) : (
                  <div className="flex items-center gap-2">
                    <LogIn className="w-4 h-4" />
                    Sign In
                  </div>
                )}
              </Button>
            </form>

            <div className="mt-6 text-center">
              <p className="text-slate-400 text-sm">
                Don't have an account?{" "}
                <Link href="/auth/signup" className="text-blue-400 hover:text-blue-300 font-medium">
                  Sign up
                </Link>
              </p>
            </div>
          </CardContent>
        </Card>
      </div>
    </>
  )
}