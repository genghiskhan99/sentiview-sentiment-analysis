"use client"

import { PieChart, Pie, Cell, ResponsiveContainer, Legend, Tooltip } from "recharts"

interface SentimentData {
  name: string
  value: number
  color: string
}

interface SentimentPieProps {
  data: {
    positive: number
    negative: number
    neutral: number
  }
  size?: "sm" | "md" | "lg"
}

export function SentimentPie({ data, size = "md" }: SentimentPieProps) {
  const total = data.positive + data.negative + data.neutral

  const chartData: SentimentData[] = [
    {
      name: "Positive",
      value: data.positive,
      color: "#22c55e", // green-500
    },
    {
      name: "Neutral",
      value: data.neutral,
      color: "#6b7280", // gray-500
    },
    {
      name: "Negative",
      value: data.negative,
      color: "#ef4444", // red-500
    },
  ].filter((item) => item.value > 0)

  const sizeConfig = {
    sm: { width: 200, height: 200, innerRadius: 40, outerRadius: 80 },
    md: { width: 300, height: 300, innerRadius: 60, outerRadius: 120 },
    lg: { width: 400, height: 400, innerRadius: 80, outerRadius: 160 },
  }

  const config = sizeConfig[size]

  const CustomTooltip = ({ active, payload }: any) => {
    if (active && payload && payload.length) {
      const data = payload[0]
      const percentage = total > 0 ? ((data.value / total) * 100).toFixed(1) : "0"
      return (
        <div className="bg-background border rounded-lg p-3 shadow-lg">
          <p className="font-medium">{data.name}</p>
          <p className="text-sm text-muted-foreground">
            {data.value} ({percentage}%)
          </p>
        </div>
      )
    }
    return null
  }

  const CustomLabel = ({ cx, cy, midAngle, innerRadius, outerRadius, percent }: any) => {
    if (percent < 0.05) return null // Don't show labels for very small slices

    const RADIAN = Math.PI / 180
    const radius = innerRadius + (outerRadius - innerRadius) * 0.5
    const x = cx + radius * Math.cos(-midAngle * RADIAN)
    const y = cy + radius * Math.sin(-midAngle * RADIAN)

    return (
      <text
        x={x}
        y={y}
        fill="white"
        textAnchor={x > cx ? "start" : "end"}
        dominantBaseline="central"
        className="text-sm font-medium"
      >
        {`${(percent * 100).toFixed(0)}%`}
      </text>
    )
  }

  if (total === 0) {
    return (
      <div className="flex items-center justify-center" style={{ width: config.width, height: config.height }}>
        <p className="text-muted-foreground">No data to display</p>
      </div>
    )
  }

  return (
    <ResponsiveContainer width="100%" height={config.height}>
      <PieChart>
        <Pie
          data={chartData}
          cx="50%"
          cy="50%"
          labelLine={false}
          label={CustomLabel}
          outerRadius={config.outerRadius}
          innerRadius={config.innerRadius}
          fill="#8884d8"
          dataKey="value"
        >
          {chartData.map((entry, index) => (
            <Cell key={`cell-${index}`} fill={entry.color} />
          ))}
        </Pie>
        <Tooltip content={<CustomTooltip />} />
        <Legend
          verticalAlign="bottom"
          height={36}
          formatter={(value, entry) => (
            <span style={{ color: entry.color }} className="text-sm">
              {value}
            </span>
          )}
        />
      </PieChart>
    </ResponsiveContainer>
  )
}
