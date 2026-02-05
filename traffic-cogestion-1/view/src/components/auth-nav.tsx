"use client"

import { useEffect, useState } from "react"
import Link from "next/link"
import { Button } from "@/components/ui/button"
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuLabel,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu"
import { Avatar, AvatarFallback, AvatarImage } from "@/components/ui/avatar"
import { LogOut, User, Settings, LayoutDashboard } from "lucide-react"

export default function AuthNav() {
  const [user, setUser] = useState<any>(null)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    // Check local storage for mocked session
    const checkSession = () => {
      try {
        const storedUser = localStorage.getItem('traffic_user')
        if (storedUser) {
          setUser(JSON.parse(storedUser))
        } else {
          setUser(null)
        }
      } catch (e) {
        console.error("Error reading user from storage", e)
      } finally {
        setLoading(false)
      }
    }

    checkSession()

    // Listen for storage events (though mostly useful for multi-tab)
    window.addEventListener('storage', checkSession)
    return () => window.removeEventListener('storage', checkSession)
  }, [])

  const handleSignOut = () => {
    localStorage.removeItem('traffic_user')
    setUser(null)
    window.location.href = '/'
  }

  if (loading) {
    return <div className="h-9 w-20 animate-pulse bg-muted rounded"></div>
  }

  if (user) {
    const initials = user.user_metadata?.full_name
      ? user.user_metadata.full_name
        .split(" ")
        .map((n: string) => n[0])
        .join("")
        .toUpperCase()
        .slice(0, 2)
      : user.email?.slice(0, 2).toUpperCase()

    const isAdmin = user.email === 'admin@traffic.com' || user.role === 'admin'

    return (
      <DropdownMenu>
        <DropdownMenuTrigger asChild>
          <Button variant="ghost" className="relative h-9 w-9 rounded-full">
            <Avatar className="h-9 w-9">
              <AvatarImage src="" alt={user.email} />
              <AvatarFallback className="bg-primary/10 text-primary">
                {initials}
              </AvatarFallback>
            </Avatar>
          </Button>
        </DropdownMenuTrigger>
        <DropdownMenuContent className="w-56" align="end" forceMount>
          <DropdownMenuLabel className="font-normal">
            <div className="flex flex-col space-y-1">
              <p className="text-sm font-medium leading-none">
                {user.user_metadata?.full_name || "User"}
              </p>
              <p className="text-xs leading-none text-muted-foreground">
                {user.email}
              </p>
              {isAdmin && (
                <span className="text-[10px] bg-blue-500/10 text-blue-500 px-1.5 py-0.5 rounded w-fit mt-1">
                  ADMIN
                </span>
              )}
            </div>
          </DropdownMenuLabel>
          <DropdownMenuSeparator />
          <DropdownMenuItem asChild>
            <Link href="/dashboard" className="cursor-pointer">
              <LayoutDashboard className="mr-2 h-4 w-4" />
              <span>Dashboard</span>
            </Link>
          </DropdownMenuItem>
          <DropdownMenuItem disabled>
            <User className="mr-2 h-4 w-4" />
            <span>Profile</span>
          </DropdownMenuItem>
          <DropdownMenuItem disabled>
            <Settings className="mr-2 h-4 w-4" />
            <span>Settings</span>
          </DropdownMenuItem>
          <DropdownMenuSeparator />
          <DropdownMenuItem onClick={handleSignOut} className="text-red-500 hover:text-red-600 cursor-pointer">
            <LogOut className="mr-2 h-4 w-4" />
            <span>Log out</span>
          </DropdownMenuItem>
        </DropdownMenuContent>
      </DropdownMenu>
    )
  }

  return (
    <div className="flex items-center gap-2">
      <Link href="/auth/signin">
        <Button variant="ghost" size="sm">
          Sign In
        </Button>
      </Link>
      <Link href="/auth/signup">
        <Button size="sm">Sign Up</Button>
      </Link>
    </div>
  )
}