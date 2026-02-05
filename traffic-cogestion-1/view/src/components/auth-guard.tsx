"use client"

import { useEffect, useState } from "react"
import { usePathname } from "next/navigation"

export function AuthGuard({ children }: { children: React.ReactNode }) {
    const pathname = usePathname()
    const [authorized, setAuthorized] = useState(false)

    // Default to true to allow hydration without blocking, 
    // but we'll effect check immediately. 
    // Actually default false + loading state is better for security, 
    // but hydration mismatch might occur if we return different server/client.
    // Since we use 'use client', it's fine.
    const [checking, setChecking] = useState(true)

    useEffect(() => {
        // List of public paths
        const publicPaths = ['/auth/signin', '/auth/signup']

        // Redirect if already logged in and trying to access signin
        if (pathname === '/auth/signin') {
            const user = localStorage.getItem('traffic_user')
            if (user) {
                window.location.href = '/dashboard'
                return
            }
        }

        // Check if current path is public
        // We allow public paths AND anything starting with /_next (assets)
        if (publicPaths.includes(pathname) || pathname === '/' || pathname?.startsWith('/_next') || pathname?.startsWith('/static')) {
            setAuthorized(true)
            setChecking(false)
            return
        }

        // Check auth
        // We use a small delay to prevent rapid state flips if localStorage read is instant but effect scheduling varies
        const checkAuth = () => {
            const user = localStorage.getItem('traffic_user')
            if (user) {
                setAuthorized(true)
                setChecking(false)
            } else {
                setAuthorized(false)
                setChecking(false)
                // Redirect to signin
                window.location.href = '/auth/signin'
            }
        }

        checkAuth()
    }, [pathname])

    if (checking) {
        // Loading screen
        return (
            <div className="min-h-screen bg-slate-950 flex flex-col items-center justify-center gap-4 text-white">
                <div className="w-8 h-8 border-4 border-blue-500/30 border-t-blue-500 rounded-full animate-spin"></div>
                <p className="text-slate-400 text-sm">Verifying authentication...</p>
            </div>
        )
    }

    // If public path or authorized, render children
    if (authorized) {
        return <>{children}</>
    }

    // If not authorized and check is done, we are redirecting.
    // Render nothing or shell.
    return null
}
