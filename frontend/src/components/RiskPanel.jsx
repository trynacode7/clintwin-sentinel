import { motion, AnimatePresence } from 'framer-motion';
import { useEffect, useState, useRef } from 'react';

/**
 * RiskPanel — Large animated risk score with status indicator
 * 
 * Shows the current deterioration probability as an animated gauge
 * with a pulsing glow effect that intensifies with risk level.
 * Status badges: Stable (green), Warning (amber), Critical (red)
 */
export default function RiskPanel({ riskScore = 0, status = 'Stable', trend = 'stable' }) {
  const [displayScore, setDisplayScore] = useState(0);
  const displayScoreRef = useRef(0);

  // Keep the ref in sync with state
  useEffect(() => {
    displayScoreRef.current = displayScore;
  }, [displayScore]);

  // Animate score counting up — uses ref to avoid stale closure
  useEffect(() => {
    const duration = 1500;
    const steps = 60;
    const startValue = displayScoreRef.current;
    const increment = (riskScore - startValue) / steps;
    let current = startValue;
    let step = 0;

    const timer = setInterval(() => {
      step++;
      current += increment;
      const clamped = Math.min(Math.max(current, 0), 1);
      setDisplayScore(clamped);
      displayScoreRef.current = clamped;
      if (step >= steps) {
        setDisplayScore(riskScore);
        displayScoreRef.current = riskScore;
        clearInterval(timer);
      }
    }, duration / steps);

    return () => clearInterval(timer);
  }, [riskScore]);

  const statusColor = {
    Stable: { ring: '#10b981', bg: 'rgba(16, 185, 129, 0.1)', text: '#34d399' },
    Warning: { ring: '#f59e0b', bg: 'rgba(245, 158, 11, 0.1)', text: '#fbbf24' },
    Critical: { ring: '#ef4444', bg: 'rgba(239, 68, 68, 0.1)', text: '#f87171' },
  }[status] || { ring: '#10b981', bg: 'rgba(16, 185, 129, 0.1)', text: '#34d399' };

  const trendIcon = trend === 'increasing' ? '↑' : trend === 'decreasing' ? '↓' : '→';
  const trendColor = trend === 'increasing' ? '#f87171' : trend === 'decreasing' ? '#34d399' : '#94a3b8';

  // Circular gauge properties
  const radius = 80;
  const circumference = 2 * Math.PI * radius;
  const progress = circumference - (displayScore * circumference);

  return (
    <motion.div
      className={`glass-panel p-6 flex flex-col items-center justify-center relative overflow-hidden risk-glow-${status.toLowerCase()}`}
      initial={{ opacity: 0, scale: 0.9 }}
      animate={{ opacity: 1, scale: 1 }}
      transition={{ duration: 0.6, ease: 'easeOut' }}
    >
      {/* Background gradient based on status */}
      <div
        className="absolute inset-0 opacity-20 pointer-events-none"
        style={{
          background: `radial-gradient(ellipse at center, ${statusColor.bg}, transparent 70%)`,
        }}
      />

      {/* Title */}
      <h2 className="text-sm font-semibold text-slate-400 uppercase tracking-wider mb-4 relative z-10">
        Deterioration Risk
      </h2>

      {/* Circular Gauge */}
      <div className="relative w-48 h-48 mb-4">
        <svg className="w-full h-full transform -rotate-90" viewBox="0 0 200 200">
          {/* Background ring */}
          <circle
            cx="100" cy="100" r={radius}
            fill="none"
            stroke="rgba(148, 163, 184, 0.1)"
            strokeWidth="12"
          />
          {/* Progress ring */}
          <motion.circle
            cx="100" cy="100" r={radius}
            fill="none"
            stroke={statusColor.ring}
            strokeWidth="12"
            strokeLinecap="round"
            strokeDasharray={circumference}
            strokeDashoffset={progress}
            initial={{ strokeDashoffset: circumference }}
            animate={{ strokeDashoffset: progress }}
            transition={{ duration: 1.5, ease: 'easeOut' }}
            style={{
              filter: `drop-shadow(0 0 8px ${statusColor.ring})`,
            }}
          />
        </svg>

        {/* Score text */}
        <div className="absolute inset-0 flex flex-col items-center justify-center">
          <AnimatePresence mode="wait">
            <motion.span
              key={Math.round(displayScore * 100)}
              className="text-5xl font-bold font-mono"
              style={{ color: statusColor.text }}
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -10 }}
              transition={{ duration: 0.3 }}
            >
              {(displayScore * 100).toFixed(0)}
            </motion.span>
          </AnimatePresence>
          <span className="text-xs text-slate-500 font-medium mt-1">/ 100</span>
        </div>
      </div>

      {/* Status Badge */}
      <motion.div
        className={`status-badge status-${status.toLowerCase()} mb-3`}
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ delay: 0.3 }}
      >
        {status}
      </motion.div>

      {/* Trend */}
      <div className="flex items-center gap-2 text-sm">
        <span style={{ color: trendColor }} className="text-lg font-bold">
          {trendIcon}
        </span>
        <span className="text-slate-400">
          {trend === 'increasing' ? 'Rising' : trend === 'decreasing' ? 'Declining' : 'Stable'}
        </span>
      </div>
    </motion.div>
  );
}
