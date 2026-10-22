import { motion } from 'framer-motion';
import { useEffect, useState, useRef } from 'react';

/**
 * RiskPanel - Large animated risk score with status indicator
 * 
 * Shows the current deterioration probability as an animated gauge
 * with a pulsing glow effect that intensifies with risk level.
 * Status badges: Stable (green), Warning (amber), Critical (red)
 */
export default function RiskPanel({ riskScore, status = 'Stable', trend = 'stable' }) {
  // Check if riskScore is genuinely available and valid
  const isScoreValid = typeof riskScore === 'number' && !isNaN(riskScore);

  // Normalize scale: if given as percentage > 1 (e.g. 48.9), scale down to 0-1 probability
  const targetScore = isScoreValid
    ? (riskScore > 1 ? riskScore / 100 : Math.max(0, Math.min(1, riskScore)))
    : 0;

  const [displayScore, setDisplayScore] = useState(targetScore);
  const displayScoreRef = useRef(targetScore);

  // Keep the ref in sync with state
  useEffect(() => {
    displayScoreRef.current = displayScore;
  }, [displayScore]);

  // Animate score counting up or down – smoothly interpolates displayScore
  useEffect(() => {
    if (!isScoreValid) return;

    const duration = 1000;
    const steps = 30;
    const startValue = displayScoreRef.current;
    const increment = (targetScore - startValue) / steps;
    let current = startValue;
    let step = 0;

    const timer = setInterval(() => {
      step++;
      current += increment;
      const clamped = Math.min(Math.max(current, 0), 1);
      setDisplayScore(clamped);
      displayScoreRef.current = clamped;
      if (step >= steps) {
        setDisplayScore(targetScore);
        displayScoreRef.current = targetScore;
        clearInterval(timer);
      }
    }, duration / steps);

    return () => clearInterval(timer);
  }, [targetScore, isScoreValid]);

  const effectiveStatus = isScoreValid ? status : 'Unavailable';

  const statusColor = {
    Stable: { ring: '#10b981', bg: 'rgba(16, 185, 129, 0.1)', text: '#34d399' },
    Warning: { ring: '#f59e0b', bg: 'rgba(245, 158, 11, 0.1)', text: '#fbbf24' },
    Critical: { ring: '#ef4444', bg: 'rgba(239, 68, 68, 0.1)', text: '#f87171' },
    Unavailable: { ring: '#475569', bg: 'rgba(71, 85, 105, 0.1)', text: '#94a3b8' },
  }[effectiveStatus] || { ring: '#10b981', bg: 'rgba(16, 185, 129, 0.1)', text: '#34d399' };

  const trendIcon = trend === 'increasing' ? '↑' : trend === 'decreasing' ? '↓' : '→';
  const trendColor = trend === 'increasing' ? '#f87171' : trend === 'decreasing' ? '#34d399' : '#94a3b8';

  // Circular gauge properties
  const radius = 80;
  const circumference = 2 * Math.PI * radius;
  const progress = isScoreValid
    ? circumference - (displayScore * circumference)
    : circumference; // Empty ring when unavailable

  return (
    <motion.div
      className={`glass-panel p-6 flex flex-col items-center justify-center relative overflow-hidden risk-glow-${effectiveStatus.toLowerCase()}`}
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
            transition={{ duration: 1.0, ease: 'easeOut' }}
            style={{
              filter: isScoreValid ? `drop-shadow(0 0 8px ${statusColor.ring})` : 'none',
            }}
          />
        </svg>

        {/* Score text */}
        <div className="absolute inset-0 flex flex-col items-center justify-center text-center p-2">
          {isScoreValid ? (
            <>
              <span
                className="text-5xl font-bold font-mono tracking-tight"
                style={{ color: statusColor.text }}
              >
                {(displayScore * 100).toFixed(0)}
              </span>
              <span className="text-xs text-slate-500 font-medium mt-1">/ 100</span>
            </>
          ) : (
            <>
              <span className="text-sm font-bold text-slate-400">Risk Unavailable</span>
              <span className="text-[10px] text-slate-500 mt-0.5">No prediction</span>
            </>
          )}
        </div>
      </div>

      {/* Status Badge */}
      <motion.div
        className={`status-badge status-${effectiveStatus.toLowerCase()} mb-3`}
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ delay: 0.3 }}
      >
        {isScoreValid ? status : 'Risk Unavailable'}
      </motion.div>

      {/* Trend */}
      {isScoreValid ? (
        <div className="flex items-center gap-2 text-sm">
          <span style={{ color: trendColor }} className="text-lg font-bold">
            {trendIcon}
          </span>
          <span className="text-slate-400">
            {trend === 'increasing' ? 'Rising' : trend === 'decreasing' ? 'Declining' : 'Stable'}
          </span>
        </div>
      ) : (
        <div className="text-xs text-slate-500">
          Awaiting vitals stream
        </div>
      )}
    </motion.div>
  );
}
