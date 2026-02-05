"use client"

import Link from "next/link"
import { Button } from "@/components/ui/button"
import { ArrowRight, Activity, Map, Shield } from "lucide-react"

export default function LandingPage() {
    return (
        <div className="flex flex-col min-h-[calc(100vh-4rem)]">
            <main className="flex-1 flex flex-col items-center justify-center p-6 text-center space-y-12 bg-slate-950 text-white overflow-hidden relative">

                {/* Abstract Background */}
                <div className="absolute inset-0 pointer-events-none overflow-hidden">
                    <div className="absolute top-1/4 left-1/4 w-96 h-96 bg-blue-500/10 rounded-full blur-3xl animate-pulse"></div>
                    <div className="absolute bottom-1/4 right-1/4 w-96 h-96 bg-purple-500/10 rounded-full blur-3xl animate-pulse" style={{ animationDelay: '2s' }}></div>
                </div>

                <div className="space-y-6 max-w-3xl relative z-10">
                    <h1 className="text-5xl md:text-7xl font-bold tracking-tight bg-clip-text text-transparent bg-gradient-to-r from-blue-400 via-purple-400 to-pink-400">
                        Smart Traffic<br />Prediction
                    </h1>
                    <p className="text-xl md:text-2xl text-slate-400 max-w-2xl mx-auto leading-relaxed">
                        AI-powered real-time congestion analysis and route optimization for smarter cities.
                    </p>
                </div>

                <div className="flex flex-col sm:flex-row gap-4 relative z-10">
                    <Link href="/auth/signin">
                        <Button size="lg" className="h-12 px-8 text-lg rounded-full bg-blue-600 hover:bg-blue-700 shadow-lg shadow-blue-900/20">
                            Get Started <ArrowRight className="ml-2 w-5 h-5" />
                        </Button>
                    </Link>
                    <Link href="/route-planner">
                        <Button size="lg" variant="outline" className="h-12 px-8 text-lg rounded-full border-slate-700 text-slate-300 hover:bg-slate-800 hover:text-white">
                            View Demo
                        </Button>
                    </Link>
                </div>

                {/* Feature Grid */}
                <div className="grid grid-cols-1 md:grid-cols-3 gap-8 mt-12 w-full max-w-5xl text-left relative z-10">
                    <div className="p-6 rounded-2xl bg-slate-900/50 border border-slate-800 backdrop-blur-sm">
                        <div className="w-12 h-12 rounded-xl bg-blue-500/20 flex items-center justify-center mb-4 text-blue-400">
                            <Activity className="w-6 h-6" />
                        </div>
                        <h3 className="text-xl font-semibold mb-2">Real-time Analysis</h3>
                        <p className="text-slate-400">Monitor traffic flow and congestion levels instantly with high-precision data.</p>
                    </div>
                    <div className="p-6 rounded-2xl bg-slate-900/50 border border-slate-800 backdrop-blur-sm">
                        <div className="w-12 h-12 rounded-xl bg-purple-500/20 flex items-center justify-center mb-4 text-purple-400">
                            <Map className="w-6 h-6" />
                        </div>
                        <h3 className="text-xl font-semibold mb-2">Smart Routing</h3>
                        <p className="text-slate-400">Find the optimal path with predictive algorithms that anticipate jams.</p>
                    </div>
                    <div className="p-6 rounded-2xl bg-slate-900/50 border border-slate-800 backdrop-blur-sm">
                        <div className="w-12 h-12 rounded-xl bg-pink-500/20 flex items-center justify-center mb-4 text-pink-400">
                            <Shield className="w-6 h-6" />
                        </div>
                        <h3 className="text-xl font-semibold mb-2">Secure & Reliable</h3>
                        <p className="text-slate-400">Enterprise-grade security ensuring your location data is confined and safe.</p>
                    </div>
                </div>

            </main>
        </div>
    )
}
