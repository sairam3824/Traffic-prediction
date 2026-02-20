"use client"

import { useState, useEffect } from "react"
import Link from "next/link"
import { usePathname } from "next/navigation"
import { cn } from "@/lib/utils"
import { Menu, X, LayoutDashboard, Map, Shield } from "lucide-react"

export default function MobileNav() {
  const [open, setOpen] = useState(false)
  const pathname = usePathname()
  const [user, setUser] = useState<any>(null)

  useEffect(() => {
    const stored = localStorage.getItem('traffic_user')
    if (stored) {
      try {
        setUser(JSON.parse(stored))
      } catch (e) {
        // ignore
      }
    }
  }, [])

  // Close menu on route change
  useEffect(() => {
    setOpen(false)
  }, [pathname])

  const isAdmin = user?.email === 'admin@traffic.com' || user?.role === 'admin'

  const routes = [
    { href: "/dashboard", label: "Dashboard", active: pathname === "/dashboard", icon: LayoutDashboard, show: true },
    { href: "/route-planner", label: "Route Planner", active: pathname === "/route-planner", icon: Map, show: true },
    { href: "/admin", label: "Admin Panel", active: pathname?.startsWith("/admin"), icon: Shield, show: isAdmin },
  ]

  return (
    <>
      <button
        onClick={() => setOpen(!open)}
        className="p-2 rounded-md text-foreground hover:bg-accent transition-colors"
        aria-label="Toggle menu"
      >
        {open ? <X className="w-5 h-5" /> : <Menu className="w-5 h-5" />}
      </button>

      {open && (
        <>
          <div className="fixed inset-0 top-[53px] bg-black/50 z-40" onClick={() => setOpen(false)} />
          <div className="fixed left-0 right-0 top-[53px] z-50 bg-background border-b border-border shadow-lg animate-in slide-in-from-top-2 duration-200">
            <div className="px-4 py-3 space-y-1">
              {routes.filter(r => r.show).map((route) => (
                <Link
                  key={route.href}
                  href={route.href}
                  className={cn(
                    "flex items-center gap-3 px-3 py-3 rounded-lg text-sm font-medium transition-colors",
                    route.active
                      ? "bg-accent text-foreground"
                      : "text-muted-foreground hover:bg-accent/50 hover:text-foreground"
                  )}
                >
                  <route.icon className="w-5 h-5" />
                  {route.label}
                </Link>
              ))}
            </div>
          </div>
        </>
      )}
    </>
  )
}
