import { motion } from 'framer-motion';
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid,
  Tooltip, ResponsiveContainer, ReferenceLine, Area, ComposedChart
} from 'recharts';
import { useState } from 'react';

/**
 * VitalsTimeline — Multi-vital line charts with threshold references
 * 
 * Displays heart_rate, MAP, SpO2 as interactive line charts.
 * Includes clinical threshold reference lines and smooth transitions.
 */

const VITAL_CONFIG = {
  heart_rate: {
    label: 'Heart Rate',
    unit: 'bpm',
    color: '#f43f5e',
    threshold: { value: 100, label: 'Tachycardia', color: '#fca5a5' },
    domain: [40, 160],
  },
  map: {
    label: 'MAP',
    unit: 'mmHg',
    color: '#3b82f6',
    threshold: { value: 65, label: 'Hypotension', color: '#93c5fd' },
    domain: [30, 120],
  },
  spo2: {
    label: 'SpO₂',
    unit: '%',
    color: '#10b981',
    threshold: { value: 92, label: 'Hypoxemia', color: '#6ee7b7' },
    domain: [80, 100],
  },
};

function CustomTooltip({ active, payload, label }) {
  if (!active || !payload?.length) return null;

  return (
    <div className="chart-tooltip">
      <p className="text-xs text-slate-400 mb-1">{label}</p>
      {payload.map((entry, i) => (
        <p key={i} className="text-sm font-medium" style={{ color: entry.color }}>
          {entry.value?.toFixed(1)} {VITAL_CONFIG[entry.dataKey]?.unit || ''}
        </p>
      ))}
    </div>
  );
}

function VitalChart({ data, vitalKey, config }) {
  return (
    <div className="h-44">
      <div className="flex items-center justify-between mb-2 px-1">
        <div className="flex items-center gap-2">
          <div className="w-2.5 h-2.5 rounded-full" style={{ backgroundColor: config.color }} />
          <span className="text-xs font-semibold text-slate-300">{config.label}</span>
        </div>
        <span className="text-xs text-slate-500">{config.unit}</span>
      </div>
      <ResponsiveContainer width="100%" height="85%">
        <ComposedChart data={data} margin={{ top: 5, right: 10, left: -20, bottom: 0 }}>
          <defs>
            <linearGradient id={`gradient-${vitalKey}`} x1="0" y1="0" x2="0" y2="1">
              <stop offset="5%" stopColor={config.color} stopOpacity={0.2} />
              <stop offset="95%" stopColor={config.color} stopOpacity={0} />
            </linearGradient>
          </defs>
          <CartesianGrid strokeDasharray="3 3" stroke="rgba(148, 163, 184, 0.08)" />
          <XAxis
            dataKey="time"
            tick={{ fontSize: 9, fill: '#64748b' }}
            tickLine={false}
            axisLine={{ stroke: 'rgba(148, 163, 184, 0.1)' }}
            interval={Math.floor(data.length / 6)}
          />
          <YAxis
            domain={config.domain}
            tick={{ fontSize: 9, fill: '#64748b' }}
            tickLine={false}
            axisLine={{ stroke: 'rgba(148, 163, 184, 0.1)' }}
            width={35}
          />
          <Tooltip content={<CustomTooltip />} />
          {config.threshold && (
            <ReferenceLine
              y={config.threshold.value}
              stroke={config.threshold.color}
              strokeDasharray="4 4"
              strokeOpacity={0.5}
              label={{
                value: config.threshold.label,
                position: 'insideTopRight',
                fill: config.threshold.color,
                fontSize: 9,
                opacity: 0.7,
              }}
            />
          )}
          <Area
            type="monotone"
            dataKey={vitalKey}
            fill={`url(#gradient-${vitalKey})`}
            stroke="none"
          />
          <Line
            type="monotone"
            dataKey={vitalKey}
            stroke={config.color}
            strokeWidth={2}
            dot={false}
            activeDot={{ r: 4, fill: config.color, strokeWidth: 0 }}
          />
        </ComposedChart>
      </ResponsiveContainer>
    </div>
  );
}

export default function VitalsTimeline({ vitals }) {
  const [activeVital, setActiveVital] = useState(null);

  if (!vitals?.timestamps) return null;

  // Transform data for Recharts
  const chartData = vitals.timestamps.map((t, i) => ({
    time: t,
    heart_rate: vitals.heart_rate?.[i],
    map: vitals.map?.[i],
    spo2: vitals.spo2?.[i],
    resp_rate: vitals.resp_rate?.[i],
    lactate: vitals.lactate?.[i],
  }));

  return (
    <motion.div
      className="glass-panel p-5"
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5, delay: 0.1 }}
    >
      <div className="flex items-center justify-between mb-4">
        <h2 className="text-sm font-semibold text-slate-400 uppercase tracking-wider">
          Vitals Timeline
        </h2>
        <div className="flex gap-2">
          {Object.entries(VITAL_CONFIG).map(([key, cfg]) => (
            <button
              key={key}
              onClick={() => setActiveVital(activeVital === key ? null : key)}
              className={`px-2.5 py-1 rounded-md text-xs font-medium transition-all duration-200 ${
                activeVital === key || !activeVital
                  ? 'opacity-100'
                  : 'opacity-40 hover:opacity-70'
              }`}
              style={{
                background: `rgba(${key === 'heart_rate' ? '244,63,94' : key === 'map' ? '59,130,246' : '16,185,129'}, 0.15)`,
                color: cfg.color,
              }}
            >
              {cfg.label}
            </button>
          ))}
        </div>
      </div>

      <div className="space-y-3">
        {Object.entries(VITAL_CONFIG).map(([key, cfg]) => (
          (!activeVital || activeVital === key) && (
            <motion.div
              key={key}
              initial={{ opacity: 0, height: 0 }}
              animate={{ opacity: 1, height: 'auto' }}
              exit={{ opacity: 0, height: 0 }}
              transition={{ duration: 0.3 }}
            >
              <VitalChart data={chartData} vitalKey={key} config={cfg} />
            </motion.div>
          )
        ))}
      </div>
    </motion.div>
  );
}
