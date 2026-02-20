"use client"

import Link from "next/link"
import { Button } from "@/components/ui/button"
import { ArrowRight, Activity, Map, Shield } from "lucide-react"
import { ProjectInfo } from "@/components/project-info"

export default function LandingPage() {
    return (
        <div className="flex flex-col min-h-[calc(100vh-4rem)]">
            <main className="flex-1 flex flex-col items-center pt-10 sm:pt-20 pb-8 sm:pb-12 px-4 sm:px-6 text-center bg-slate-950 text-white relative min-h-screen">

                {/* Abstract Background */}
                <div className="absolute inset-0 pointer-events-none overflow-hidden">
                    <div className="absolute top-1/4 left-1/4 w-48 sm:w-96 h-48 sm:h-96 bg-blue-500/10 rounded-full blur-3xl animate-pulse"></div>
                    <div className="absolute bottom-1/4 right-1/4 w-48 sm:w-96 h-48 sm:h-96 bg-purple-500/10 rounded-full blur-3xl animate-pulse" style={{ animationDelay: '2s' }}></div>
                </div>

                <div className="w-full max-w-7xl relative z-10 flex flex-col lg:flex-row items-center justify-center gap-8 sm:gap-12 lg:gap-24 mb-12 sm:mb-20">
                    {/* Left Side - Project Info Card */}
                    <div className="w-full lg:w-auto flex justify-center lg:justify-end">
                        <ProjectInfo />
                    </div>

                    {/* Right Side - Hero Content */}
                    <div className="space-y-8 text-center lg:text-left max-w-2xl">
                        <div className="space-y-4">
                            <div className="inline-flex py-1 px-3 rounded-full bg-blue-500/10 border border-blue-500/20 text-blue-400 text-sm font-medium mb-2">
                                <span className="flex h-2 w-2 rounded-full bg-blue-400 mr-2 self-center animate-pulse"></span>
                                AI-Powered Traffic Analysis
                            </div>
                            <h1 className="text-3xl sm:text-5xl md:text-7xl font-bold tracking-tight bg-clip-text text-transparent bg-gradient-to-r from-blue-400 via-purple-400 to-pink-400 leading-tight">
                                Smart Traffic<br />Prediction
                            </h1>
                            <p className="text-base sm:text-xl text-slate-400 leading-relaxed">
                                AI-powered real-time congestion analysis and route optimization for smarter cities.
                            </p>
                        </div>

                        <div className="flex flex-col sm:flex-row gap-4 justify-center lg:justify-start">
                            <Link href="/auth/signin">
                                <Button size="lg" className="h-12 px-8 text-lg rounded-full bg-blue-600 hover:bg-blue-700 shadow-lg shadow-blue-900/20 w-full sm:w-auto">
                                    Get Started <ArrowRight className="ml-2 w-5 h-5" />
                                </Button>
                            </Link>
                            <Link href="/route-planner">
                                <Button size="lg" variant="outline" className="h-12 px-8 text-lg rounded-full border-slate-700 text-slate-300 hover:bg-slate-800 hover:text-white w-full sm:w-auto">
                                    View Demo
                                </Button>
                            </Link>
                        </div>
                    </div>
                </div>

                {/* Feature Grid */}
                <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 gap-4 sm:gap-8 w-full max-w-6xl text-left relative z-10 px-0 sm:px-4 mb-12 sm:mb-20">
                    <div className="p-6 rounded-2xl bg-slate-900/50 border border-slate-800 backdrop-blur-sm hover:bg-slate-900/80 transition-colors">
                        <div className="w-12 h-12 rounded-xl bg-blue-500/20 flex items-center justify-center mb-4 text-blue-400">
                            <Activity className="w-6 h-6" />
                        </div>
                        <h3 className="text-xl font-semibold mb-2 text-white">Real-time Analysis</h3>
                        <p className="text-slate-400">Monitor traffic flow and congestion levels instantly with high-precision data.</p>
                    </div>
                    <div className="p-6 rounded-2xl bg-slate-900/50 border border-slate-800 backdrop-blur-sm hover:bg-slate-900/80 transition-colors">
                        <div className="w-12 h-12 rounded-xl bg-purple-500/20 flex items-center justify-center mb-4 text-purple-400">
                            <Map className="w-6 h-6" />
                        </div>
                        <h3 className="text-xl font-semibold mb-2 text-white">Smart Routing</h3>
                        <p className="text-slate-400">Find the optimal path with predictive algorithms that anticipate jams.</p>
                    </div>
                    <div className="p-6 rounded-2xl bg-slate-900/50 border border-slate-800 backdrop-blur-sm hover:bg-slate-900/80 transition-colors">
                        <div className="w-12 h-12 rounded-xl bg-pink-500/20 flex items-center justify-center mb-4 text-pink-400">
                            <Shield className="w-6 h-6" />
                        </div>
                        <h3 className="text-xl font-semibold mb-2 text-white">Secure & Reliable</h3>
                        <p className="text-slate-400">Enterprise-grade security ensuring your location data is confined and safe.</p>
                    </div>
                </div>



                {/* Bottom Gradient */}
                <div className="absolute bottom-0 left-0 right-0 h-[500px] bg-gradient-to-t from-blue-950/20 via-slate-950/50 to-transparent pointer-events-none"></div>

                {/* Footer Section */}
                <footer className="w-full max-w-7xl mx-auto px-4 sm:px-6 relative z-10 border-t border-slate-800/50 pt-6 sm:pt-8 mt-auto">
                    <div className="flex flex-col md:flex-row items-center justify-between gap-3 sm:gap-4 text-slate-500 text-xs sm:text-sm">
                        <p>© 2024 Traffic Prediction. All rights reserved.</p>
                        <div className="flex items-center gap-6">
                            <Link href="#" className="hover:text-blue-400 transition-colors">Privacy Policy</Link>
                            <Link href="#" className="hover:text-blue-400 transition-colors">Terms of Service</Link>
                            <Link href="#" className="hover:text-blue-400 transition-colors">Contact</Link>
                        </div>
                    </div>
                </footer>

            </main>
        </div>
    )
}
