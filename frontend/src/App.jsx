import { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { usePatientData } from './hooks/usePatientData';
import RiskPanel from './components/RiskPanel';
import VitalsTimeline from './components/VitalsTimeline';
import LatentStateViz from './components/LatentStateViz';
import AttributionPanel from './components/AttributionPanel';
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
          const patientId = patient.id ?? patient.stay_id;
          const isSelected = patientId === selectedId;
          const status = patient.status || patient.latest_status || "Stable";
          const riskScore = patient.risk_score ?? patient.latest_risk_score ?? 0;
          const hours = patient.hours_in_icu ?? patient.n_observed_timesteps ?? 0;
          const patientName = patient.name || (patientId ? "Patient " + patientId : "Patient");
          const statusColor = {
            Stable: "#10b981",
            Warning: "#f59e0b",
            Critical: "#ef4444",
          }[status] || "#94a3b8";

          return (
            <motion.button
              key={patientId}
              onClick={() => onSelect(patientId)}
              className={`w-full flex items-center gap-3 px-3 py-2.5 rounded-xl text-left transition-all duration-200 ${
                isSelected
                  ? "bg-slate-800/80 border border-slate-700"
                  : "hover:bg-slate-800/40 border border-transparent"
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
                  {patientName}
                </p>
                <p className="text-[10px] text-slate-500">
                  {hours.toFixed(0)}h • Risk: {(riskScore * 100).toFixed(1)}%
                </p>
              </div>
              <div
                className={`status-badge text-[9px] px-1.5 py-0.5 status-${status.toLowerCase()}`}
              >
                {status.charAt(0)}
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


function MethodologyModal({ isOpen, onClose }) {
  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/80 backdrop-blur-sm">
      <motion.div
        className="w-full max-w-2xl glass-panel p-6 border border-slate-700 bg-slate-950 shadow-2xl rounded-2xl max-h-[85vh] overflow-y-auto"
        initial={{ opacity: 0, scale: 0.95 }}
        animate={{ opacity: 1, scale: 1 }}
        exit={{ opacity: 0, scale: 0.95 }}
      >
        <div className="flex items-center justify-between pb-3 border-b border-slate-800">
          <div className="flex items-center gap-2.5">
            <div className="w-8 h-8 rounded-lg bg-cyan-500/20 border border-cyan-500/30 flex items-center justify-center">
              <svg className="w-4 h-4 text-cyan-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
              </svg>
            </div>
            <div>
              <h3 className="text-sm font-bold text-white">System Architecture & Research Methodology</h3>
              <p className="text-[11px] text-slate-400">ClinTwin Sentinel Decision-Support Intelligence</p>
            </div>
          </div>
          <button
            onClick={onClose}
            className="p-1.5 rounded-lg text-slate-400 hover:text-white hover:bg-slate-800 transition-colors"
          >
            <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
            </svg>
          </button>
        </div>

        <div className="mt-4 space-y-3.5 text-xs">
          <div className="p-3.5 rounded-xl bg-slate-900/80 border border-slate-800 space-y-1.5">
            <div className="flex items-center gap-2">
              <span className="w-2 h-2 rounded-full bg-cyan-400" />
              <h4 className="font-semibold text-slate-200">1. Sepsis Risk Modeling (BiLSTM)</h4>
            </div>
            <p className="text-slate-400 leading-relaxed text-[11px]">
              Continuous ICU deterioration risk scores are computed via a bidirectional recurrent neural network (BiLSTM) processing longitudinal patient trajectories across 46 physiological vitals and laboratory features. Hidden state representations (256-dimensional z_t) capture temporal trajectory trends.
            </p>
          </div>

          <div className="p-3.5 rounded-xl bg-slate-900/80 border border-slate-800 space-y-1.5">
            <div className="flex items-center gap-2">
              <span className="w-2 h-2 rounded-full bg-purple-400" />
              <h4 className="font-semibold text-slate-200">2. Feature Attribution (Input × Gradient)</h4>
            </div>
            <p className="text-slate-400 leading-relaxed text-[11px]">
              Risk factor attributions are derived using input-gradient saliency attribution (x × ∇_x y), highlighting key directional physiological drivers pushing toward higher risk (red) or baseline stability (blue).
            </p>
          </div>

          <div className="p-3.5 rounded-xl bg-slate-900/80 border border-slate-800 space-y-1.5">
            <div className="flex items-center gap-2">
              <span className="w-2 h-2 rounded-full bg-emerald-400" />
              <h4 className="font-semibold text-slate-200">3. Physiological Digital Twin (BioGears Engine)</h4>
            </div>
            <p className="text-slate-400 leading-relaxed text-[11px]">
              Intervention trajectories (saline fluid resuscitation and norepinephrine titration) are simulated using the BioGears mechanistic human physiology engine. Solved through ordinary differential equations (ODEs), it models cardiovascular dynamics and volume distribution on continuous timelines.
            </p>
          </div>

          <div className="p-3.5 rounded-xl bg-slate-900/80 border border-slate-800 space-y-1.5">
            <div className="flex items-center gap-2">
              <span className="w-2 h-2 rounded-full bg-blue-400" />
              <h4 className="font-semibold text-slate-200">4. Inference Modes: Cohort Baseline vs. Dynamic ML</h4>
            </div>
            <p className="text-slate-400 leading-relaxed text-[11px]">
              <strong className="text-slate-300">Cohort Baseline:</strong> Retrieves stored model predictions precomputed across the patient admission windows during cohort dataset evaluation. <br />
              <strong className="text-slate-300">Dynamic ML Model:</strong> Executes a live forward pass via <code>POST /predict</code> using the patient's continuous 72-window trajectory, generating real-time deterioration probability and gradient-derived feature attributions. Both pathways evaluate the identical 46-feature BiLSTM architecture; live inference execution does not imply higher clinical accuracy over precomputed cohort benchmarks.
            </p>
          </div>

          <div className="p-3.5 rounded-xl bg-amber-950/20 border border-amber-900/40 text-amber-300 space-y-1 text-[11px]">
            <p className="font-semibold text-amber-200">Research & Data Governance Boundaries</p>
            <p className="text-amber-300/90 leading-relaxed">
              ClinTwin Sentinel is designed as an investigative decision-support research prototype. BioGears simulations evaluate virtual reference physiology (StandardMale.xml) to model mechanistic drug/fluid responses. They do not constitute individualized clinical treatment recommendations.
            </p>
          </div>
        </div>

        <div className="mt-5 pt-3 border-t border-slate-800 flex justify-end">
          <button
            onClick={onClose}
            className="px-4 py-1.5 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-200 text-xs font-semibold transition-all"
          >
            Close
          </button>
        </div>
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
  const [showMethodology, setShowMethodology] = useState(false);

  if (loading && !data) return <LoadingState />;

  const currentPatient = data?.current_patient || {};

  return (
    <div className="min-h-screen bg-[#0a0e1a]">
      {/* Demo-mode fallback banner (when backend is down) */}
      {demoMode && (
        <div className="bg-amber-950/40 border-b border-amber-700/40 px-6 py-2 flex items-center justify-between text-xs">
          <div className="flex items-center gap-2.5">
            <div className="w-2 h-2 rounded-full bg-amber-400 animate-pulse" />
            <span className="text-amber-300 font-medium">
              Backend Offline — FastAPI backend unreachable (health check failed). Displaying simulated client-side fallback data.
              Start backend with{' '}
              <code className="font-mono bg-amber-900/70 text-amber-200 px-1.5 py-0.5 rounded border border-amber-700/50">
                python -m backend.api
              </code>
            </span>
          </div>
          <span className="text-[10px] text-amber-400/90 font-semibold px-2 py-0.5 rounded bg-amber-900/40 border border-amber-700/50 uppercase tracking-wider">
            Simulated Fallback Mode
          </span>
        </div>
      )}

      {/* Non-fatal error banner (e.g. dynamic inference warning without marking backend offline) */}
      {error && !demoMode && (
        <div className="bg-red-950/40 border-b border-red-800/40 px-6 py-2 flex items-center justify-between text-xs text-red-300">
          <div className="flex items-center gap-2">
            <span className="w-2 h-2 rounded-full bg-red-400" />
            <span>{error}</span>
          </div>
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
                Cohort Baseline
              </button>
              <button
                onClick={() => setMode('real_ml')}
                className={`px-3 py-1 rounded-md text-xs font-semibold transition-all ${
                  mode === 'real_ml'
                    ? 'bg-purple-500/25 text-purple-300 border border-purple-500/50 shadow-sm'
                    : 'text-slate-400 hover:text-slate-200'
                }`}
              >
                Dynamic ML Model
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
                {mode === 'real_ml' ? 'Active BiLSTM Pipeline' : 'Clinical Cohort View'}
              </span>
            </div>

            {/* Methodology & Transparency Modal Trigger */}
            <button
              onClick={() => setShowMethodology(true)}
              className="px-2.5 py-1.5 rounded-lg bg-slate-800/50 hover:bg-slate-700/50 border border-slate-700/50 text-slate-300 text-xs font-medium transition-all flex items-center gap-1.5 cursor-pointer"
              title="System Architecture & Research Methodology"
            >
              <svg className="w-3.5 h-3.5 text-cyan-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
              </svg>
              <span>Methodology</span>
            </button>

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

      {/* Mode Context & Architecture Transparency Bar */}
      <div className="bg-slate-900/60 border-b border-slate-800/80 px-6 py-2 flex flex-col md:flex-row md:items-center justify-between gap-2 text-xs">
        <div className="flex items-center gap-2.5">
          <div
            className={`w-2 h-2 rounded-full ${
              demoMode
                ? 'bg-amber-400 animate-pulse'
                : mode === 'real_ml'
                ? 'bg-purple-400 animate-pulse'
                : 'bg-emerald-400'
            }`}
          />
          <div className="flex flex-wrap items-center gap-1.5">
            <span className="font-semibold text-slate-200">
              {demoMode
                ? 'Local Fallback Client:'
                : mode === 'real_ml'
                ? 'Dynamic ML Model Mode:'
                : 'Cohort Baseline Mode:'}
            </span>
            <span className="text-slate-400 text-[11px]">
              {demoMode
                ? 'Client-side synthetic data rendered because FastAPI backend is offline.'
                : mode === 'real_ml'
                ? 'Executing live forward-pass inference (POST /predict) with the 46-feature BiLSTM network on the current vital window sequence.'
                : 'Displaying precomputed model predictions evaluated over historical cohort admission windows (stored offline predictions).'}
            </span>
          </div>
        </div>
        <div className="flex items-center gap-2 text-[10px] text-slate-400 flex-shrink-0">
          <span className="px-2 py-0.5 rounded bg-slate-800/80 border border-slate-700/60 text-slate-300">
            {demoMode
              ? 'Local Fallback Data'
              : mode === 'real_ml'
              ? 'Live Model Inference'
              : 'Stored Cohort Predictions'}
          </span>
          <span className="hidden lg:inline text-slate-500">
            • Dynamic ML executes on-demand inference; does not assert higher clinical accuracy over cohort baseline.
          </span>
        </div>
      </div>

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
                riskScore={currentPatient.risk_score}
                status={currentPatient.status || 'Stable'}
                trend={currentPatient.risk_trend || 'stable'}
              />
            </div>
            <div className="col-span-12 lg:col-span-9 xl:col-span-10">
              <VitalsTimeline vitals={data?.vitals} />
            </div>

            {/* Row 2: Feature Attribution + Latent State + Attention */}
            <div className="col-span-12 lg:col-span-4">
              <AttributionPanel attributionData={data?.attribution || data?.shap} />
            </div>
            <div className="col-span-12 lg:col-span-4">
              <LatentStateViz trajectory={data?.latent_trajectory} />
            </div>
            <div className="col-span-12 lg:col-span-4">
              <AttentionHeatmap attention={data?.attention} attributionData={data?.attribution || data?.shap} />
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
              <WhatIfSimulationPanel patient={currentPatient} />
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
      <MethodologyModal isOpen={showMethodology} onClose={() => setShowMethodology(false)} />
    </div>
  );
}
