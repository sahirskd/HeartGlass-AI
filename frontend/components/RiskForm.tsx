"use client"
import React, { useState } from 'react'
import { motion } from 'framer-motion'
import { Activity, Heart, Info } from 'lucide-react'

const formFields = [
  { name: 'age', label: 'Age', type: 'number', placeholder: 'e.g. 52' },
  { name: 'sex', label: 'Sex', type: 'select', options: ['Male', 'Female'] },
  { name: 'dataset', label: 'Region/Dataset', type: 'select', options: ['Cleveland', 'Hungary', 'Switzerland', 'VA Long Beach'] },
  { name: 'cp', label: 'Chest Pain Type', type: 'select', options: ['typical angina', 'asymptomatic', 'non-anginal', 'atypical angina'] },
  { name: 'trestbps', label: 'Resting Blood Pressure (mm Hg)', type: 'number', placeholder: 'e.g. 120' },
  { name: 'chol', label: 'Cholesterol (mg/dl)', type: 'number', placeholder: 'e.g. 200' },
  { name: 'fbs', label: 'Fasting Blood Sugar > 120 mg/dl', type: 'select', options: ['TRUE', 'FALSE'] },
  { name: 'restecg', label: 'Resting ECG', type: 'select', options: ['lv hypertrophy', 'normal', 'st-t abnormality'] },
  { name: 'thalch', label: 'Max Heart Rate', type: 'number', placeholder: 'e.g. 150' },
  { name: 'exang', label: 'Exercise Induced Angina', type: 'select', options: ['TRUE', 'FALSE'] },
  { name: 'oldpeak', label: 'ST Depression', type: 'number', placeholder: 'e.g. 1.5' },
  { name: 'slope', label: 'ST Slope', type: 'select', options: ['downsloping', 'flat', 'upsloping'] },
  { name: 'ca', label: 'Major Vessels (0-3)', type: 'number', placeholder: '0' },
  { name: 'thal', label: 'Thalassemia', type: 'select', options: ['fixed defect', 'normal', 'reversable defect'] },
]

export default function RiskForm({ onResult }: { onResult: (res: any) => void }) {
  const [formData, setFormData] = useState<any>({
    age: 50, sex: 'Male', dataset: 'Cleveland', cp: 'asymptomatic',
    trestbps: 120, chol: 200, fbs: 'FALSE', restecg: 'normal',
    thalch: 150, exang: 'FALSE', oldpeak: 0, slope: 'flat',
    ca: 0, thal: 'normal'
  })
  const [loading, setLoading] = useState(false)

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    setLoading(true)
    try {
      const res = await fetch('/api/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(formData)
      })
      const predictData = await res.json()
      
      const explainRes = await fetch('/api/explain', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(formData)
      })
      const explainData = await explainRes.json()
      
      onResult({ ...predictData, ...explainData })
    } catch (err) {
      console.error(err)
    } finally {
      setLoading(false)
    }
  }

  const handleChange = (e: any) => {
    const { name, value, type } = e.target
    setFormData((prev: any) => ({
      ...prev,
      [name]: type === 'number' ? parseFloat(value) : value
    }))
  }

  return (
    <motion.div 
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      className="glass p-8 rounded-3xl w-full max-w-4xl shadow-2xl"
    >
      <div className="flex items-center gap-4 mb-8">
        <div className="p-3 bg-primary/20 rounded-2xl">
          <Activity className="text-primary" size={32} />
        </div>
        <div>
          <h2 className="text-2xl font-bold">Patient Assessment</h2>
          <p className="text-slate-400 text-sm">Enter clinical parameters for risk scoring</p>
        </div>
      </div>

      <form onSubmit={handleSubmit} className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {formFields.map((field) => (
          <div key={field.name} className="flex flex-col gap-2">
            <label className="text-xs font-semibold text-slate-400 uppercase tracking-wider px-1">
              {field.label}
            </label>
            {field.type === 'select' ? (
              <select
                name={field.name}
                value={formData[field.name]}
                onChange={handleChange}
                className="bg-slate-800/50 border border-slate-700/50 rounded-xl px-4 py-3 focus:ring-2 focus:ring-primary outline-none transition-all appearance-none text-sm"
              >
                {field.options?.map(opt => (
                  <option key={opt} value={opt}>{opt}</option>
                ))}
              </select>
            ) : (
              <input
                type="number"
                name={field.name}
                value={formData[field.name]}
                onChange={handleChange}
                step="any"
                placeholder={field.placeholder}
                className="bg-slate-800/50 border border-slate-700/50 rounded-xl px-4 py-3 focus:ring-2 focus:ring-primary outline-none transition-all text-sm"
              />
            )}
          </div>
        ))}

        <div className="md:col-span-2 pt-4">
          <button
            type="submit"
            disabled={loading}
            className="w-full bg-primary hover:bg-primary-dark text-white font-bold py-4 rounded-2xl transition-all flex items-center justify-center gap-2 disabled:bg-slate-700 disabled:cursor-not-allowed group"
          >
            {loading ? (
              <div className="w-5 h-5 border-2 border-white/30 border-t-white rounded-full animate-spin" />
            ) : (
              <>
                <Heart className="group-hover:scale-110 transition-transform" size={20} />
                Generate Risk Score
              </>
            )}
          </button>
        </div>
      </form>
    </motion.div>
  )
}
