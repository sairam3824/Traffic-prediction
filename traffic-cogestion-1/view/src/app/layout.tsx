import type React from "react"
import type { Metadata, Viewport } from "next"
import { Geist, Geist_Mono } from "next/font/google"
import { ThemeProvider } from "@/components/theme-provider"
import AuthNav from "@/components/auth-nav"
import MainNav from "@/components/main-nav"
import MobileNav from "@/components/mobile-nav"
import { AuthGuard } from "@/components/auth-guard"
import "./globals.css"

const geistSans = Geist({
  variable: "--font-geist-sans",
  subsets: ["latin"],
})

const geistMono = Geist_Mono({
  variable: "--font-geist-mono",
  subsets: ["latin"],
})

export const metadata: Metadata = {
  title: "Traffic Prediction (Demo)",
  description: "Traffic Prediction and Congestion Analysis Demo",
}

export const viewport: Viewport = {
  width: "device-width",
  initialScale: 1,
  maximumScale: 1,
  userScalable: false,
  viewportFit: "cover",
}

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode
}>) {
  return (
    <html lang="en" suppressHydrationWarning>
      <body className={`${geistSans.variable} ${geistMono.variable} font-sans antialiased`}>
        <ThemeProvider
          attribute="class"
          defaultTheme="dark"
          enableSystem={false}
          forcedTheme="dark"
        >
          <AuthGuard>
            <nav className="bg-background border-b border-border sticky top-0 z-50 backdrop-blur-sm">
              <div className="max-w-7xl mx-auto px-4 sm:px-6 py-3 sm:py-4 flex items-center justify-between">
                <div className="flex items-center gap-4 sm:gap-8">
                  <a href="/" className="text-foreground font-bold text-base sm:text-lg hover:text-primary transition-colors">
                    Traffic Prediction
                  </a>
                  <div className="hidden md:block">
                    <MainNav />
                  </div>
                </div>
                <div className="flex items-center gap-2 sm:gap-4">
                  <AuthNav />
                  <div className="md:hidden">
                    <MobileNav />
                  </div>
                </div>
              </div>
            </nav>
            {children}
          </AuthGuard>
        </ThemeProvider>
      </body>
    </html>
  )
}
