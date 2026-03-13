"use client"

import React from 'react'
import { motion } from 'framer-motion'
import { TrendingUp, TrendingDown, AlertCircle, ShieldCheck, Info } from 'lucide-react'
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Cell } from 'recharts'

interface Factor {
  feature: string
  impact: number
}

interface ResultData {
  probability: number
  risk_level: string
  top_factors: Factor[]
}

export default function ResultDashboard({ data }: { data: ResultData }) {
  const percentage = Math.round(data.probability * 100)
  const isHighRisk = data.risk_level === "High"

  return (
    <motion.div 
      initial={{ opacity: 0, scale: 0.95 }}
      animate={{ opacity: 1, scale: 1 }}
      className="glass p-8 rounded-3xl w-full max-w-4xl shadow-2xl mt-12 mb-20"
    >
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-12">
        {/* Score Circle */}
        <div className="flex flex-col items-center justify-center space-y-4">
          <div className="relative w-48 h-48">
            <svg className="w-full h-full transform -rotate-90">
              <circle
                cx="96" cy="96" r="80"
                fill="transparent"
                stroke="currentColor"
                strokeWidth="12"
                className="text-slate-800"
              />
              <motion.circle
                cx="96" cy="96" r="80"
                fill="transparent"
                stroke="currentColor"
                strokeWidth="12"
                strokeDasharray={2 * 3.14159 * 80}
                initial={{ strokeDashoffset: 2 * 3.14159 * 80 }}
                animate={{ strokeDashoffset: 2 * 3.14159 * 80 * (1 - data.probability) }}
                transition={{ duration: 1.5, ease: "easeOut" }}
                strokeLinecap="round"
                className={isHighRisk ? "text-rose-500" : "text-emerald-500"}
              />
            </svg>
            <div className="absolute inset-0 flex flex-col items-center justify-center leading-tight">
              <span className="text-5xl font-black">{percentage}%</span>
              <span className="text-slate-400 text-xs font-bold uppercase tracking-widest">Risk Score</span>
            </div>
          </div>
          <div className={`px-4 py-1.5 rounded-full text-sm font-bold flex items-center gap-2 ${
            isHighRisk ? "bg-rose-500/10 text-rose-500" : "bg-emerald-500/10 text-emerald-500"
          }`}>
            {isHighRisk ? <AlertCircle size={16} /> : <ShieldCheck size={16} />}
            {data.risk_level} Risk Level
          </div>
        </div>

        {/* Explainability Section */}
        <div className="lg:col-span-2 space-y-6">
          <div>
            <h3 className="text-xl font-bold mb-1">Risk Factors Analysis</h3>
            <p className="text-slate-400 text-sm">Key clinical features contributing to this score (SHAP values)</p>
          </div>

          <div className="space-y-4">
            {data.top_factors.map((factor, idx) => {
              const impactPercent = Math.min(Math.abs(factor.impact) * 200, 100)
              const isPositive = factor.impact > 0

              return (
                <div key={idx} className="space-y-1">
                  <div className="flex justify-between text-sm">
                    <span className="font-semibold text-slate-300">{factor.feature}</span>
                    <span className={`flex items-center gap-1 font-mono text-xs ${isPositive ? 'text-rose-400' : 'text-emerald-400'}`}>
                      {isPositive ? <TrendingUp size={12} /> : <TrendingDown size={12} />}
                      {factor.impact.toFixed(3)}
                    </span>
                  </div>
                  <div className="h-2 bg-slate-800 rounded-full overflow-hidden">
                    <motion.div 
                      initial={{ width: 0 }}
                      animate={{ width: `${impactPercent}%` }}
                      transition={{ duration: 1, delay: 0.5 + idx * 0.1 }}
                      className={`h-full ${isPositive ? 'bg-rose-500' : 'bg-emerald-500'}`}
                    />
                  </div>
                </div>
              )
            })}
          </div>

          <div className="p-4 bg-slate-800/30 rounded-2xl border border-slate-700/30">
            <div className="flex gap-3">
              <Info className="text-primary shrink-0" size={20} />
              <p className="text-xs text-slate-400 leading-relaxed italic">
                Red bars indicate features increasing the heart disease risk, while green bars indicate protective factors. 
                SHAP analysis reveals how each feature pushes the score from the baseline.
              </p>
            </div>
          </div>
        </div>
      </div>
    </motion.div>
  )
}
