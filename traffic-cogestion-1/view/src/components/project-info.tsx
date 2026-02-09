"use client"

import { User, Mail, Phone, Globe, Linkedin, Github } from "lucide-react"

export function ProjectInfo() {
    return (
        <div className="w-full max-w-md bg-slate-900/50 backdrop-blur-sm border border-slate-800 rounded-2xl p-6 shadow-xl overflow-hidden">
            <div className="flex items-center gap-4 mb-6">
                <div className="h-12 w-12 rounded-full bg-cyan-500/10 flex items-center justify-center text-cyan-400 shrink-0 ring-1 ring-cyan-500/20">
                    <User size={24} />
                </div>
                <div className="text-left">
                    <h4 className="font-bold text-lg text-white">Project Built By</h4>
                    <p className="text-sm text-slate-400">Developed by Sairam Maruri. Let's connect!</p>
                </div>
            </div>

            <div className="flex flex-col gap-4">
                <div className="flex items-center gap-3 text-sm group/item">
                    <User size={18} className="text-cyan-400 shrink-0" />
                    <span className="font-medium text-slate-300">MARURI SAI RAMA LINGA REDDY</span>
                </div>

                <a
                    href="mailto:sairam.maruri@gmail.com"
                    className="flex items-center gap-3 text-sm group/item hover:text-cyan-400 transition-colors text-slate-300"
                >
                    <Mail size={18} className="text-cyan-400 shrink-0" />
                    <span>sairam.maruri@gmail.com</span>
                </a>

                <a
                    href="tel:+917893865644"
                    className="flex items-center gap-3 text-sm group/item hover:text-cyan-400 transition-colors text-slate-300"
                >
                    <Phone size={18} className="text-cyan-400 shrink-0" />
                    <span>+91 78938 65644</span>
                </a>

                <a
                    href="https://saiii.in"
                    target="_blank"
                    rel="noopener noreferrer"
                    className="flex items-center gap-3 text-sm group/item hover:text-cyan-400 transition-colors text-slate-300"
                >
                    <Globe size={18} className="text-cyan-400 shrink-0" />
                    <span>Portfolio - saiii.in</span>
                </a>

                <a
                    href="https://linkedin.com/in/sairam-maruri"
                    target="_blank"
                    rel="noopener noreferrer"
                    className="flex items-center gap-3 text-sm group/item hover:text-cyan-400 transition-colors text-slate-300"
                >
                    <Linkedin size={18} className="text-cyan-400 shrink-0" />
                    <span>linkedin.com/in/sairam-maruri/</span>
                </a>

                <a
                    href="https://github.com/sairam3824"
                    target="_blank"
                    rel="noopener noreferrer"
                    className="flex items-center gap-3 text-sm group/item hover:text-cyan-400 transition-colors text-slate-300"
                >
                    <Github size={18} className="text-cyan-400 shrink-0" />
                    <span>github.com/sairam3824</span>
                </a>
            </div>
        </div>
    )
}
