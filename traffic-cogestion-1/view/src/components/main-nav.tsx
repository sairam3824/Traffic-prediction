"use client"

import Link from "next/link"
import { usePathname } from "next/navigation"
import { cn } from "@/lib/utils"
import { Shield, Clock, Map, LayoutDashboard } from "lucide-react"
import { useEffect, useState } from "react"

export default function MainNav() {
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

  // Admin check
  const isAdmin = user?.email === 'admin@traffic.com' || user?.role === 'admin'

  const routes = [
    {
      href: "/dashboard",
      label: "Dashboard",
      active: pathname === "/dashboard",
      icon: LayoutDashboard,
      show: true
    },
    {
      href: "/route-planner",
      label: "Route Planner",
      active: pathname === "/route-planner",
      icon: Map,
      show: true // Always show route planner
    },
    {
      href: "/admin",
      label: "Admin Panel",
      active: pathname?.startsWith("/admin"),
      icon: Shield,
      show: isAdmin
    }
  ]

  return (
    <nav className="flex items-center space-x-4 lg:space-x-6">
      {routes.filter(r => r.show).map((route) => (
        <Link
          key={route.href}
          href={route.href}
          className={cn(
            "text-sm font-medium transition-colors hover:text-primary flex items-center gap-2",
            route.active
              ? "text-foreground"
              : "text-muted-foreground"
          )}
        >
          {route.icon && <route.icon className="w-4 h-4" />}
          {route.label}
        </Link>
      ))}
    </nav>
  )
}