"use client"

import { useEffect } from "react"
import { useRouter } from "next/navigation"

export default function SignUpPage() {
  const router = useRouter()

  useEffect(() => {
    // Redirect to signin immediately as per request
    router.replace("/auth/signin")
  }, [router])

  return (
    <div className="min-h-screen bg-background flex items-center justify-center">
      <div className="text-slate-400">Redirecting to Sign In...</div>
    </div>
  )
}