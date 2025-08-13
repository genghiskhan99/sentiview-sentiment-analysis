"use client"

import { useState, useEffect } from "react"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { SentimentPie } from "./sentiment-pie"
import { SentimentTimeline } from "./sentiment-timeline"

interface ReviewData {
  id: string
  text: string
  label: "positive" | "negative" | "neutral"
  score: number
  created_at: string
}

interface LiveChartsProps {
  reviews: ReviewData[]
  isLive?: boolean
}

export function LiveCharts({ reviews, isLive = false }: LiveChartsProps) {
  const [chartData, setChartData] = useState({
    pie: { positive: 0, negative: 0, neutral: 0 },
    timeline: [] as Array<{
      timestamp: string
      time: string
      positive: number
      negative: number
      neutral: number
    }>,
  })

  useEffect(() => {
    // Calculate pie chart data
    const sentimentCounts = reviews.reduce(
      (acc, review) => {
        acc[review.label]++
        return acc
      },
      { positive: 0, negative: 0, neutral: 0 },
    )

    // Generate timeline data (group by time intervals)
    const timelineMap = new Map()
    reviews.forEach((review) => {
      const date = new Date(review.created_at)
      const timeKey = `${date.getHours()}:${date.getMinutes().toString().padStart(2, "0")}`

      if (!timelineMap.has(timeKey)) {
        timelineMap.set(timeKey, {
          timestamp: review.created_at,
          time: timeKey,
          positive: 0,
          negative: 0,
          neutral: 0,
        })
      }

      const entry = timelineMap.get(timeKey)
      entry[review.label]++
    })

    const timelineData = Array.from(timelineMap.values()).sort(
      (a, b) => new Date(a.timestamp).getTime() - new Date(b.timestamp).getTime(),
    )

    setChartData({
      pie: sentimentCounts,
      timeline: timelineData,
    })
  }, [reviews])

  const totalReviews = reviews.length
  const positivePercentage = totalReviews > 0 ? ((chartData.pie.positive / totalReviews) * 100).toFixed(1) : "0"
  const neutralPercentage = totalReviews > 0 ? ((chartData.pie.neutral / totalReviews) * 100).toFixed(1) : "0"
  const negativePercentage = totalReviews > 0 ? ((chartData.pie.negative / totalReviews) * 100).toFixed(1) : "0"

  return (
    <div className="space-y-6">
      {/* Summary Cards */}
      <div className="grid gap-4 md:grid-cols-4">
        <Card>
          <CardContent className="p-6">
            <div className="text-center">
              <div className="text-2xl font-bold">{totalReviews}</div>
              <p className="text-sm text-muted-foreground">Total Reviews</p>
            </div>
          </CardContent>
        </Card>
        <Card>
          <CardContent className="p-6">
            <div className="text-center">
              <div className="text-2xl font-bold text-green-600">{positivePercentage}%</div>
              <p className="text-sm text-muted-foreground">Positive</p>
            </div>
          </CardContent>
        </Card>
        <Card>
          <CardContent className="p-6">
            <div className="text-center">
              <div className="text-2xl font-bold text-gray-600">{neutralPercentage}%</div>
              <p className="text-sm text-muted-foreground">Neutral</p>
            </div>
          </CardContent>
        </Card>
        <Card>
          <CardContent className="p-6">
            <div className="text-center">
              <div className="text-2xl font-bold text-red-600">{negativePercentage}%</div>
              <p className="text-sm text-muted-foreground">Negative</p>
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Charts */}
      <div className="grid gap-6 lg:grid-cols-2">
        <Card>
          <CardHeader>
            <CardTitle>Sentiment Distribution</CardTitle>
            <CardDescription>{isLive ? "Live sentiment breakdown" : "Overall sentiment breakdown"}</CardDescription>
          </CardHeader>
          <CardContent>
            <SentimentPie data={chartData.pie} size="md" />
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>Timeline</CardTitle>
            <CardDescription>{isLive ? "Real-time sentiment flow" : "Sentiment over time"}</CardDescription>
          </CardHeader>
          <CardContent>
            <SentimentTimeline data={chartData.timeline} height={300} />
          </CardContent>
        </Card>
      </div>
    </div>
  )
}
