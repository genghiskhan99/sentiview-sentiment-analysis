"use client"

import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { SentimentPie } from "./sentiment-pie"
import { SentimentTimeline } from "./sentiment-timeline"
import { WordCloudView } from "./word-cloud-view"

interface SentimentResult {
  label: "positive" | "negative" | "neutral"
  score: number
  tokens: Array<{
    term: string
    weight: number
  }>
}

interface SentimentDashboardProps {
  result: SentimentResult
  sessionHistory?: Array<{
    timestamp: string
    sentiment: "positive" | "negative" | "neutral"
    score: number
  }>
}

export function SentimentDashboard({ result, sessionHistory = [] }: SentimentDashboardProps) {
  // Generate pie chart data from current result
  const pieData = {
    positive: result.label === "positive" ? 1 : 0,
    negative: result.label === "negative" ? 1 : 0,
    neutral: result.label === "neutral" ? 1 : 0,
  }

  // Generate timeline data from session history
  const timelineData = sessionHistory.map((item, index) => ({
    timestamp: item.timestamp,
    time: new Date(item.timestamp).toLocaleTimeString("en-US", {
      hour: "2-digit",
      minute: "2-digit",
    }),
    positive: item.sentiment === "positive" ? 1 : 0,
    negative: item.sentiment === "negative" ? 1 : 0,
    neutral: item.sentiment === "neutral" ? 1 : 0,
  }))

  return (
    <div className="grid gap-6 lg:grid-cols-2">
      {/* Sentiment Distribution */}
      <Card>
        <CardHeader>
          <CardTitle>Sentiment Distribution</CardTitle>
          <CardDescription>Overall emotional tone breakdown</CardDescription>
        </CardHeader>
        <CardContent>
          <SentimentPie data={pieData} size="md" />
        </CardContent>
      </Card>

      {/* Word Cloud */}
      <Card>
        <CardHeader>
          <CardTitle>Key Terms</CardTitle>
          <CardDescription>Most influential words in your text</CardDescription>
        </CardHeader>
        <CardContent>
          <WordCloudView words={result.tokens} height={300} />
        </CardContent>
      </Card>

      {/* Timeline (only show if there's session history) */}
      {timelineData.length > 1 && (
        <Card className="lg:col-span-2">
          <CardHeader>
            <CardTitle>Session Timeline</CardTitle>
            <CardDescription>Sentiment analysis history for this session</CardDescription>
          </CardHeader>
          <CardContent>
            <SentimentTimeline data={timelineData} height={250} />
          </CardContent>
        </Card>
      )}
    </div>
  )
}
