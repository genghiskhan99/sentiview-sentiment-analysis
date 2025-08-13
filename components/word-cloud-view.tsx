"use client"

import { useMemo } from "react"

interface WordData {
  text: string
  value: number
  sentiment?: "positive" | "negative" | "neutral"
}

interface WordCloudViewProps {
  words?: Array<{
    term: string
    weight: number
  }> | null
  height?: number
  maxWords?: number
}

export function WordCloudView({ words = [], height = 300, maxWords = 50 }: WordCloudViewProps) {
  const wordCloudData = useMemo(() => {
    if (!words || !Array.isArray(words) || words.length === 0) {
      return []
    }

    try {
      const processedWords = words
        .filter((word) => {
          return (
            word &&
            typeof word === "object" &&
            typeof word.term === "string" &&
            word.term.trim().length > 0 &&
            typeof word.weight === "number" &&
            !isNaN(word.weight)
          )
        })
        .map((word) => {
          const sentiment = word.weight > 0.1 ? "positive" : word.weight < -0.1 ? "negative" : "neutral"
          return {
            text: word.term.trim(),
            value: Math.abs(word.weight) * 100 + 20, // Scale for better visual representation
            sentiment,
          }
        })
        .sort((a, b) => b.value - a.value)
        .slice(0, maxWords)

      return processedWords
    } catch (error) {
      console.error("Error processing word cloud data:", error)
      return []
    }
  }, [words, maxWords])

  if (!wordCloudData || wordCloudData.length === 0) {
    return (
      <div className="flex items-center justify-center border rounded-lg bg-muted/20" style={{ height }}>
        <p className="text-muted-foreground">No words to display</p>
      </div>
    )
  }

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-center gap-4 text-sm">
        <div className="flex items-center gap-1">
          <div className="w-3 h-3 rounded-full bg-green-500"></div>
          <span className="text-muted-foreground">Positive</span>
        </div>
        <div className="flex items-center gap-1">
          <div className="w-3 h-3 rounded-full bg-red-500"></div>
          <span className="text-muted-foreground">Negative</span>
        </div>
        <div className="flex items-center gap-1">
          <div className="w-3 h-3 rounded-full bg-gray-500"></div>
          <span className="text-muted-foreground">Neutral</span>
        </div>
      </div>

      <div
        className="flex flex-wrap items-center justify-center gap-3 p-6 border rounded-lg bg-gradient-to-br from-muted/20 to-muted/40 overflow-hidden"
        style={{ height }}
      >
        {wordCloudData.map((word, index) => {
          const fontSize = Math.max(14, Math.min(36, word.value / 3))
          const opacity = Math.max(0.7, Math.min(1, word.value / 100))

          // Sentiment-based colors
          const colorClass =
            word.sentiment === "positive"
              ? "text-green-600 hover:text-green-700 bg-green-50 hover:bg-green-100 border-green-200"
              : word.sentiment === "negative"
                ? "text-red-600 hover:text-red-700 bg-red-50 hover:bg-red-100 border-red-200"
                : "text-gray-600 hover:text-gray-700 bg-gray-50 hover:bg-gray-100 border-gray-200"

          return (
            <div
              key={`${word.text}-${index}`}
              className={`
                inline-block px-3 py-1.5 rounded-lg border font-medium 
                transition-all duration-200 cursor-pointer transform hover:scale-105 hover:shadow-sm
                ${colorClass}
              `}
              style={{
                fontSize: `${fontSize}px`,
                opacity,
              }}
              title={`Weight: ${words?.[index]?.weight?.toFixed(3) || "N/A"}`}
            >
              {word.text}
            </div>
          )
        })}
      </div>
    </div>
  )
}
