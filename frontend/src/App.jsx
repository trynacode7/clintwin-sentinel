import { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { usePatientData } from './hooks/usePatientData';
import RiskPanel from './components/RiskPanel';
import VitalsTimeline from './components/VitalsTimeline';
import LatentStateViz from './components/LatentStateViz';
import ShapPanel from './components/ShapPanel';
import AttentionHeatmap from './components/AttentionHeatmap';
import ClinicalExplanation from './components/ClinicalExplanation';
import WhatIfSimulationPanel from './components/WhatIfSimulationPanel';

/**
 * App — ICU Digital Twin Dashboard
 * 
 * Main layout with 6 panels in a responsive grid:
 * ┌─────────────┬────────────────────────────┐
 * │ Risk Panel  │     Vitals Timeline        │
 * ├─────────────┼──────────────┬─────────────┤
 * │ SHAP Panel  │ Latent State │  Attention  │
 * ├─────────────┴──────────────┴─────────────┤
 * │         Clinical Explanation              │
 * └──────────────────────────────────────────┘
 */

function PatientSidebar({ patients, selectedId, onSelect }) {
  return (
    <div className="glass-panel p-3 overflow-y-auto max-h-[calc(100vh-120px)]">
      <h3 className="text-xs font-semibold text-slate-500 uppercase tracking-wider mb-3 px-2">
        ICU Patients
      </h3>
      <div className="space-y-1">
        {patients?.map((patient) => {
          const isSelected = patient.id === selectedId;
          const statusColor = {
            Stable: '#10b981',
            Warning: '#f59e0b',
            Critical: '#ef4444',
          }[patient.status] || '#94a3b8';

          return (
            <motion.button
              key={patient.id}
              onClick={() => onSelect(patient.id)}
              className={`w-full flex items-center gap-3 px-3 py-2.5 rounded-xl text-left transition-all duration-200 ${
                isSelected
                  ? 'bg-slate-800/80 border border-slate-700'
                  : 'hover:bg-slate-800/40 border border-transparent'
              }`}
              whileHover={{ x: 2 }}
              whileTap={{ scale: 0.98 }}
            >
              <div
                className="w-2 h-2 rounded-full flex-shrink-0"
                style={{ backgroundColor: statusColor, boxShadow: `0 0 6px ${statusColor}` }}
              />
              <div className="flex-1 min-w-0">
                <p className="text-xs font-medium text-slate-300 truncate">
                  {patient.name}
                </p>
                <p className="text-[10px] text-slate-500">
                  {patient.hours_in_icu?.toFixed(0)}h · Risk: {(patient.risk_score * 100).toFixed(0)}%
                </p>
              </div>
              <div
                className={`status-badge text-[9px] px-1.5 py-0.5 status-${patient.status.toLowerCase()}`}
              >
                {patient.status.charAt(0)}
              </div>
            </motion.button>
          );
        })}
      </div>
    </div>
  );
}

function LoadingState() {
  return (
    <div className="flex items-center justify-center h-screen bg-[#0a0e1a]">
      <motion.div
        className="text-center"
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
      >
        <motion.div
          className="w-16 h-16 border-4 border-cyan-500/30 border-t-cyan-500 rounded-full mx-auto mb-4"
          animate={{ rotate: 360 }}
          transition={{ duration: 1, repeat: Infinity, ease: 'linear' }}
        />
        <p className="text-slate-400 text-sm">Initializing Digital Twin...</p>
      </motion.div>
    </div>
  );
}

export default function App() {
  const {
    data,
    loading,
    error,
    demoMode,
    mode,
    setMode,
    patientLoading,
    selectedPatient,
    setSelectedPatient,
    refresh,
  } = usePatientData();
  const [sidebarOpen, setSidebarOpen] = useState(true);

  if (loading && !data) return <LoadingState />;

  const currentPatient = data?.current_patient || {};

  return (
    <div className="min-h-screen bg-[#0a0e1a]">
      {/* Demo-mode fallback banner (when backend is down) */}
      {demoMode && (
        <div className="bg-amber-900/30 border-b border-amber-700/30 px-6 py-1.5 flex items-center gap-2">
          <div className="w-1.5 h-1.5 rounded-full bg-amber-400 animate-pulse" />
          <span className="text-xs text-amber-400 font-medium">
            Backend Offline — FastAPI backend unreachable. Displaying local client fallback data.
            Start the backend with{' '}
            <code className="font-mono bg-amber-900/50 px-1 rounded">
              python -m backend.api
            </code>
          </span>
        </div>
      )}

      {/* Real ML Mode checkpoint diagnostic banner */}
      {!demoMode && mode === 'real_ml' && (
        <div className="bg-purple-950/40 border-b border-purple-800/40 px-6 py-1.5 flex items-center justify-between">
          <div className="flex items-center gap-2">
            <div className="w-2 h-2 rounded-full bg-purple-400 animate-pulse" />
            <span className="text-xs text-purple-300 font-medium">
              Real ML Inference Active: Running full 46-feature preprocessing and BiLSTM latent extraction (256-dim) via POST /predict.
            </span>
          </div>
          <span className="text-[11px] text-purple-400/80 bg-purple-900/40 px-2 py-0.5 rounded border border-purple-700/40">
            BiLSTM retrained: 46 features | 256-D z_t | non-saturated risk
          </span>
        </div>
      )}

      {/* Top Header */}
      <header className="sticky top-0 z-50 glass-panel rounded-none border-x-0 border-t-0">
        <div className="flex items-center justify-between px-6 py-3">
          <div className="flex items-center gap-4">
            <motion.button
              onClick={() => setSidebarOpen(!sidebarOpen)}
              className="p-2 rounded-lg hover:bg-slate-800/50 transition-colors"
              whileTap={{ scale: 0.95 }}
            >
              <svg className="w-5 h-5 text-slate-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 6h16M4 12h16M4 18h16" />
              </svg>
            </motion.button>
            <div className="flex items-center gap-3">
              <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-cyan-500 to-blue-600 flex items-center justify-center">
                <span className="text-white text-sm font-bold">DT</span>
              </div>
              <div>
                <h1 className="text-base font-bold gradient-text">ICU Digital Twin</h1>
                <p className="text-[10px] text-slate-500 -mt-0.5">Clinical Intelligence Dashboard</p>
              </div>
            </div>
          </div>

          <div className="flex items-center gap-4">
            {/* Mode Switcher */}
            <div className="flex items-center rounded-lg bg-slate-900/90 p-1 border border-slate-700/60 shadow-inner">
              <button
                onClick={() => setMode('demo')}
                className={`px-3 py-1 rounded-md text-xs font-semibold transition-all ${
                  mode === 'demo'
                    ? 'bg-cyan-500/20 text-cyan-300 border border-cyan-500/40 shadow-sm'
                    : 'text-slate-400 hover:text-slate-200'
                }`}
              >
                Demo Mode (/patient)
              </button>
              <button
                onClick={() => setMode('real_ml')}
                className={`px-3 py-1 rounded-md text-xs font-semibold transition-all ${
                  mode === 'real_ml'
                    ? 'bg-purple-500/25 text-purple-300 border border-purple-500/50 shadow-sm'
                    : 'text-slate-400 hover:text-slate-200'
                }`}
              >
                Real ML Mode (/predict)
              </button>
            </div>

            {/* Mode Badge Indicator */}
            <div className="hidden sm:flex items-center gap-2 px-3 py-1.5 rounded-lg bg-slate-800/50 border border-slate-700/50">
              <div
                className={`w-1.5 h-1.5 rounded-full animate-pulse ${
                  mode === 'real_ml' ? 'bg-purple-400' : 'bg-emerald-400'
                }`}
              />
              <span className="text-xs text-slate-300 font-medium">
                {mode === 'real_ml' ? 'BiLSTM Live Pipeline' : 'Demo Archetypes'}
              </span>
            </div>

            <motion.button
              onClick={refresh}
              className="p-2 rounded-lg hover:bg-slate-800/50 transition-colors"
              whileTap={{ scale: 0.9, rotate: 180 }}
              title="Refresh data"
            >
              <svg className="w-4 h-4 text-slate-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
              </svg>
            </motion.button>
          </div>
        </div>
      </header>

      <div className="flex">
        {/* Patient Sidebar */}
        <AnimatePresence>
          {sidebarOpen && (
            <motion.aside
              className="w-56 flex-shrink-0 p-3 pr-0"
              initial={{ width: 0, opacity: 0 }}
              animate={{ width: 224, opacity: 1 }}
              exit={{ width: 0, opacity: 0 }}
              transition={{ duration: 0.3 }}
            >
              <PatientSidebar
                patients={data?.patients}
                selectedId={selectedPatient}
                onSelect={setSelectedPatient}
              />
            </motion.aside>
          )}
        </AnimatePresence>

        {/* Main Dashboard Grid */}
        <main className="flex-1 p-3 overflow-y-auto relative" style={{ maxHeight: 'calc(100vh - 60px)' }}>
          {/* Per-patient loading overlay */}
          {patientLoading && (
            <div className="absolute inset-0 z-20 flex items-center justify-center bg-[#0a0e1a]/60 backdrop-blur-sm rounded-xl">
              <div className="flex flex-col items-center gap-3">
                <motion.div
                  className="w-8 h-8 border-2 border-cyan-500/30 border-t-cyan-500 rounded-full"
                  animate={{ rotate: 360 }}
                  transition={{ duration: 0.8, repeat: Infinity, ease: 'linear' }}
                />
                <p className="text-xs text-slate-400">Loading patient data…</p>
              </div>
            </div>
          )}
          <div className="grid grid-cols-12 gap-3 auto-rows-auto">
            {/* Row 1: Risk Panel + Vitals */}
            <div className="col-span-12 lg:col-span-3 xl:col-span-2">
              <RiskPanel
                riskScore={currentPatient.risk_score || 0}
                status={currentPatient.status || 'Stable'}
                trend={currentPatient.risk_trend || 'stable'}
              />
            </div>
            <div className="col-span-12 lg:col-span-9 xl:col-span-10">
              <VitalsTimeline vitals={data?.vitals} />
            </div>

            {/* Row 2: SHAP + Latent State + Attention */}
            <div className="col-span-12 lg:col-span-4">
              <ShapPanel shapData={data?.shap} />
            </div>
            <div className="col-span-12 lg:col-span-4">
              <LatentStateViz trajectory={data?.latent_trajectory} />
            </div>
            <div className="col-span-12 lg:col-span-4">
              <AttentionHeatmap attention={data?.attention} />
            </div>

            {/* Row 3: Clinical Explanation (full width) */}
            <div className="col-span-12">
              <ClinicalExplanation
                explanation={data?.explanation}
                riskScore={currentPatient.risk_score || 0}
                status={currentPatient.status || 'Stable'}
                vitals={data?.vitals}
              />
            </div>

            {/* Row 4: BioGears What-If Physiological Intervention Simulation */}
            <div className="col-span-12">
              <WhatIfSimulationPanel />
            </div>
          </div>

          {/* Footer */}
          <div className="mt-4 pb-4 text-center">
            <p className="text-[10px] text-slate-700">
              ICU Digital Twin System v1.0 • MIMIC-IV Compatible • Research Use Only
            </p>
          </div>
        </main>
      </div>
    </div>
  );
}
