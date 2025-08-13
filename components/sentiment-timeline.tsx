"use client"

import { AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend } from "recharts"

interface TimelineData {
  timestamp: string
  positive: number
  negative: number
  neutral: number
  time: string // formatted time for display
}

interface SentimentTimelineProps {
  data: TimelineData[]
  height?: number
}

export function SentimentTimeline({ data, height = 300 }: SentimentTimelineProps) {
  const CustomTooltip = ({ active, payload, label }: any) => {
    if (active && payload && payload.length) {
      return (
        <div className="bg-background border rounded-lg p-3 shadow-lg">
          <p className="font-medium mb-2">{label}</p>
          {payload.map((entry: any, index: number) => (
            <p key={index} className="text-sm" style={{ color: entry.color }}>
              {entry.name}: {entry.value}
            </p>
          ))}
        </div>
      )
    }
    return null
  }

  if (data.length === 0) {
    return (
      <div className="flex items-center justify-center" style={{ height }}>
        <p className="text-muted-foreground">No timeline data available</p>
      </div>
    )
  }

  return (
    <ResponsiveContainer width="100%" height={height}>
      <AreaChart data={data} margin={{ top: 10, right: 30, left: 0, bottom: 0 }}>
        <defs>
          <linearGradient id="colorPositive" x1="0" y1="0" x2="0" y2="1">
            <stop offset="5%" stopColor="#22c55e" stopOpacity={0.8} />
            <stop offset="95%" stopColor="#22c55e" stopOpacity={0.1} />
          </linearGradient>
          <linearGradient id="colorNeutral" x1="0" y1="0" x2="0" y2="1">
            <stop offset="5%" stopColor="#6b7280" stopOpacity={0.8} />
            <stop offset="95%" stopColor="#6b7280" stopOpacity={0.1} />
          </linearGradient>
          <linearGradient id="colorNegative" x1="0" y1="0" x2="0" y2="1">
            <stop offset="5%" stopColor="#ef4444" stopOpacity={0.8} />
            <stop offset="95%" stopColor="#ef4444" stopOpacity={0.1} />
          </linearGradient>
        </defs>
        <XAxis dataKey="time" className="text-xs" />
        <YAxis className="text-xs" />
        <CartesianGrid strokeDasharray="3 3" className="opacity-30" />
        <Tooltip content={<CustomTooltip />} />
        <Legend />
        <Area
          type="monotone"
          dataKey="positive"
          stackId="1"
          stroke="#22c55e"
          fill="url(#colorPositive)"
          name="Positive"
        />
        <Area type="monotone" dataKey="neutral" stackId="1" stroke="#6b7280" fill="url(#colorNeutral)" name="Neutral" />
        <Area
          type="monotone"
          dataKey="negative"
          stackId="1"
          stroke="#ef4444"
          fill="url(#colorNegative)"
          name="Negative"
        />
      </AreaChart>
    </ResponsiveContainer>
  )
}
