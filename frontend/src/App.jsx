import { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { usePatientData } from './hooks/usePatientData';
import RiskPanel from './components/RiskPanel';
import VitalsTimeline from './components/VitalsTimeline';
import LatentStateViz from './components/LatentStateViz';
import AttributionPanel from './components/AttributionPanel';
import AttentionHeatmap from './components/AttentionHeatmap';
import ClinicalExplanation from './components/ClinicalExplanation';
import WhatIfPanel from './components/WhatIfPanel';

/**
 * App — ICU Digital Twin Dashboard
 *
 * Phase K: wired to the real backend (see hooks/usePatientData.js) —
 * every value rendered below comes from a real trained-model prediction
 * on a real synthetic ICU stay, fetched from backend/api.py. Nothing in
 * this component or its data source fabricates a clinical value.
 *
 * Main layout with 7 panels in a responsive grid:
 * ┌─────────────┬────────────────────────────┐
 * │ Risk Panel  │     Vitals Timeline        │
 * ├─────────────┼──────────────┬─────────────┤
 * │ Attribution │ Latent State │  Attention  │
 * ├───────────────────────┬────────────────────┤
 * │ Clinical Explanation  │  What-If Scenario  │
 * └───────────────────────┴────────────────────┘
 *
 * What-If Scenario (pre-BioGears architecture — see backend/whatif.py
 * and WHATIF_ARCHITECTURE.md): lets the user submit a real, validated
 * intervention scenario and see the real baseline plus an honest
 * "simulation requires BioGears integration" result. No fabricated
 * counterfactual is ever shown.
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

function ErrorState({ message }) {
  return (
    <div className="flex items-center justify-center h-screen bg-[#0a0e1a] px-6">
      <div className="glass-panel p-6 max-w-md text-center">
        <p className="text-red-400 text-sm font-semibold mb-2">Could not load real patient data</p>
        <p className="text-slate-400 text-xs leading-relaxed">{message}</p>
        <p className="text-slate-600 text-[10px] mt-3">
          This dashboard only shows real backend data — it does not fall back to
          fabricated values. Start the backend (`python -m uvicorn backend.api:app`
          on port 8000) and reload.
        </p>
      </div>
    </div>
  );
}

export default function App() {
  const {
    data,
    loading,
    detailLoading,
    error,
    selectedPatient,
    setSelectedPatient,
    refresh,
    intervenableFeatures,
    runWhatIf,
  } = usePatientData();
  const [sidebarOpen, setSidebarOpen] = useState(true);

  if (loading && !data) return <LoadingState />;
  if (error && !data?.current_patient) return <ErrorState message={error} />;

  const currentPatient = data?.current_patient || {};
  const model = data?.model;

  return (
    <div className="min-h-screen bg-[#0a0e1a]">
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
            {model && (
              <div className="flex items-center gap-2 px-3 py-1.5 rounded-lg bg-slate-800/50 border border-slate-700/50">
                <div className={`w-1.5 h-1.5 rounded-full ${detailLoading ? 'bg-amber-400 animate-pulse' : 'bg-emerald-400'}`} />
                <span className="text-xs text-slate-400">
                  {detailLoading ? 'Loading real prediction…' : `Real ${model.type.toUpperCase()} inference (epoch ${model.epoch})`}
                </span>
              </div>
            )}
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
        <main className="flex-1 p-3 overflow-y-auto" style={{ maxHeight: 'calc(100vh - 60px)' }}>
          {error && currentPatient.id != null && (
            <div className="mb-3 px-4 py-2 rounded-lg bg-red-500/10 border border-red-500/20 text-xs text-red-300">
              Could not load the latest data for this patient ({error}) — showing the last successfully
              loaded real prediction instead.
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

            {/* Row 2: Attribution + Latent State + Attention */}
            <div className="col-span-12 lg:col-span-4">
              <AttributionPanel attributionData={data?.attribution} method={data?.attribution_method} />
            </div>
            <div className="col-span-12 lg:col-span-4">
              <LatentStateViz
                trajectory={data?.latent_trajectory}
                disclaimer={data?.twin_disclaimer}
                twinState={data?.twin_state}
              />
            </div>
            <div className="col-span-12 lg:col-span-4">
              <AttentionHeatmap attention={data?.attention} />
            </div>

            {/* Row 3: Clinical Explanation + What-If */}
            <div className="col-span-12 lg:col-span-7">
              <ClinicalExplanation
                explanation={data?.explanation}
                riskScore={currentPatient.risk_score || 0}
                status={currentPatient.status || 'Stable'}
                vitals={data?.vitals}
              />
            </div>
            <div className="col-span-12 lg:col-span-5">
              <WhatIfPanel
                intervenableFeatures={intervenableFeatures}
                onRunScenario={runWhatIf}
                stayId={currentPatient.id}
              />
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
