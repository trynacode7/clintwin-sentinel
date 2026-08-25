import { motion } from 'framer-motion';
import { useState, useEffect, useRef } from 'react';

/**
 * ClinicalExplanation — Natural language explanation panel
 * 
 * Displays AI-generated clinical explanations with typewriter animation.
 * Dynamically shows current vitals from live data.
 */

export default function ClinicalExplanation({ explanation, riskScore = 0, status = 'Stable', vitals }) {
  const [displayedText, setDisplayedText] = useState('');
  const [isTyping, setIsTyping] = useState(false);
  const prevExplanationRef = useRef('');

  // Typewriter animation — only re-trigger when explanation TEXT actually changes
  useEffect(() => {
    if (!explanation || explanation === prevExplanationRef.current) {
      // If same explanation, just show it fully (don't re-type on live tick)
      if (explanation) setDisplayedText(explanation);
      return;
    }
    
    prevExplanationRef.current = explanation;
    setIsTyping(true);
    setDisplayedText('');
    let i = 0;
    
    const timer = setInterval(() => {
      if (i < explanation.length) {
        setDisplayedText(explanation.slice(0, i + 1));
        i++;
      } else {
        setIsTyping(false);
        clearInterval(timer);
      }
    }, 18);

    return () => clearInterval(timer);
  }, [explanation]);

  const statusColors = {
    Stable: { icon: '✓', color: '#10b981', bg: 'rgba(16, 185, 129, 0.08)' },
    Warning: { icon: '⚠', color: '#f59e0b', bg: 'rgba(245, 158, 11, 0.08)' },
    Critical: { icon: '⚡', color: '#ef4444', bg: 'rgba(239, 68, 68, 0.08)' },
  };

  const statusConfig = statusColors[status] || statusColors.Stable;

  // Compute key factors from latest vitals (live data)
  const latestHR = vitals?.heart_rate?.[vitals.heart_rate.length - 1] ?? '--';
  const latestMAP = vitals?.map?.[vitals.map.length - 1] ?? '--';
  const latestLactate = vitals?.lactate?.filter(v => v != null).slice(-1)[0] ?? '--';

  const hrAlert = typeof latestHR === 'number' && latestHR > 100;
  const mapAlert = typeof latestMAP === 'number' && latestMAP < 65;
  const lacAlert = typeof latestLactate === 'number' && latestLactate > 2;

  const factors = [
    {
      label: 'Lactate',
      value: typeof latestLactate === 'number' ? `${latestLactate.toFixed(1)} mmol/L` : '-- mmol/L',
      alert: lacAlert,
      icon: '🔬',
    },
    {
      label: 'MAP',
      value: typeof latestMAP === 'number' ? `${latestMAP.toFixed(0)} mmHg` : '-- mmHg',
      alert: mapAlert,
      icon: '💓',
    },
    {
      label: 'HR',
      value: typeof latestHR === 'number' ? `${latestHR.toFixed(0)} bpm` : '-- bpm',
      alert: hrAlert,
      icon: '💗',
    },
  ];

  return (
    <motion.div
      className="glass-panel p-5"
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5, delay: 0.5 }}
    >
      <div className="flex items-center justify-between mb-4">
        <h2 className="text-sm font-semibold text-slate-400 uppercase tracking-wider">
          Clinical Interpretation
        </h2>
        <div className="flex items-center gap-2">
          <div className="w-1.5 h-1.5 rounded-full bg-cyan-400 animate-pulse" />
          <span className="text-xs text-slate-600">AI Analysis</span>
        </div>
      </div>

      {/* Status header */}
      <motion.div
        className="flex items-center gap-3 mb-4 p-3 rounded-xl"
        style={{ background: statusConfig.bg }}
      >
        <span className="text-2xl">{statusConfig.icon}</span>
        <div>
          <span
            className="text-sm font-bold"
            style={{ color: statusConfig.color }}
          >
            {status} — Risk {(riskScore * 100).toFixed(0)}%
          </span>
        </div>
      </motion.div>

      {/* Explanation text with typewriter */}
      <div
        className="p-4 rounded-xl text-sm leading-relaxed min-h-[80px]"
        style={{
          background: 'rgba(15, 23, 42, 0.5)',
          border: '1px solid rgba(148, 163, 184, 0.08)',
        }}
      >
        <span className="text-slate-300">
          {displayedText}
        </span>
        {isTyping && (
          <motion.span
            className="inline-block w-0.5 h-4 ml-0.5 align-text-bottom"
            style={{ background: '#06b6d4' }}
            animate={{ opacity: [1, 0] }}
            transition={{ duration: 0.5, repeat: Infinity }}
          />
        )}
      </div>

      {/* Key factors summary — LIVE from vitals */}
      <div className="mt-4 grid grid-cols-3 gap-2">
        {factors.map((factor, i) => (
          <motion.div
            key={factor.label}
            className="p-2.5 rounded-lg text-center"
            style={{
              background: factor.alert ? 'rgba(239, 68, 68, 0.06)' : 'rgba(15, 23, 42, 0.4)',
              border: factor.alert
                ? '1px solid rgba(239, 68, 68, 0.15)'
                : '1px solid rgba(148, 163, 184, 0.05)',
            }}
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.7 + i * 0.1 }}
          >
            <span className="text-base">{factor.icon}</span>
            <p className="text-xs text-slate-500 mt-1">{factor.label}</p>
            <p
              className="text-xs font-bold mt-0.5"
              style={{ color: factor.alert ? '#f87171' : '#94a3b8' }}
            >
              {factor.value}
            </p>
          </motion.div>
        ))}
      </div>

      {/* Timestamp */}
      <div className="mt-3 text-right">
        <span className="text-[10px] text-slate-600 font-mono">
          Analysis generated at {new Date().toLocaleTimeString()}
        </span>
      </div>
    </motion.div>
  );
}
