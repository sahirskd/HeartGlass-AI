"use client"
import RiskForm from '../components/RiskForm'
import ResultDashboard from '../components/ResultDashboard'
import { useState } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { HeartPulse } from 'lucide-react'

export default function Home() {
  const [result, setResult] = useState(null)

  return (
    <main className="flex flex-col items-center py-12 px-4 md:px-8">
      <motion.div 
        initial={{ opacity: 0, y: -20 }}
        animate={{ opacity: 1, y: 0 }}
        className="text-center mb-16 space-y-4"
      >
        <div className="flex justify-center mb-6">
          <div className="bg-primary/10 p-4 rounded-3xl animate-glow">
            <HeartPulse className="text-primary w-12 h-12" />
          </div>
        </div>
        <h1 className="text-5xl md:text-6xl font-black bg-clip-text text-transparent bg-gradient-to-r from-white via-white to-slate-500 tracking-tight">
          CardioRisk AI
        </h1>
        <p className="text-slate-400 text-lg max-w-2xl mx-auto">
          Explainable Risk Scoring System for Chronic Heart Disease using Cost-Sensitive Deep Learning Specialists.
        </p>
      </motion.div>

      <RiskForm onResult={setResult} />

      <AnimatePresence>
        {result && (
          <div id="results">
            <ResultDashboard data={result} />
          </div>
        )}
      </AnimatePresence>

      <footer className="mt-20 text-slate-500 text-xs py-12 border-t border-slate-800 w-full text-center">
        © 2026 CardioRisk Intelligence. Clinical Decision Support System.
      </footer>
    </main>
  )
}
