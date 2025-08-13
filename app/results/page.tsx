"use client"

import { useEffect, useState } from "react"
import { useRouter } from "next/navigation"
import { Button } from "@/components/ui/button"
import { ResultCards } from "@/components/result-cards"
import { SentimentDashboard } from "@/components/sentiment-dashboard"
import { Card, CardContent } from "@/components/ui/card"
import { Loader2, AlertCircle } from "lucide-react"
import { useToast } from "@/hooks/use-toast"

interface SentimentResult {
  label: "positive" | "negative" | "neutral"
  score: number
  tokens: Array<{
    term: string
    weight: number
  }>
}

export default function ResultsPage() {
  const [result, setResult] = useState<SentimentResult | null>(null)
  const [isLoading, setIsLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [analyzedText, setAnalyzedText] = useState<string>("")
  const [sessionHistory, setSessionHistory] = useState<
    Array<{
      timestamp: string
      sentiment: "positive" | "negative" | "neutral"
      score: number
    }>
  >([])
  const router = useRouter()
  const { toast } = useToast()

  useEffect(() => {
    const text = sessionStorage.getItem("analysisText")
    if (!text) {
      router.push("/")
      return
    }

    setAnalyzedText(text)
    analyzeText(text)

    // Load session history
    const history = JSON.parse(sessionStorage.getItem("sessionHistory") || "[]")
    setSessionHistory(history)
  }, [router])

  const analyzeText = async (text: string) => {
    try {
      setIsLoading(true)
      setError(null)

      // For now, use mock data since backend isn't ready
      const mockResult = generateMockResult(text)

      // Simulate API delay
      await new Promise((resolve) => setTimeout(resolve, 1500))

      setResult(mockResult)

      // Add to session history
      const newHistoryItem = {
        timestamp: new Date().toISOString(),
        sentiment: mockResult.label,
        score: mockResult.score,
      }

      const updatedHistory = [...sessionHistory, newHistoryItem]
      setSessionHistory(updatedHistory)
      sessionStorage.setItem("sessionHistory", JSON.stringify(updatedHistory))
    } catch (err) {
      setError("Failed to analyze sentiment. Please try again.")
      toast({
        title: "Analysis Failed",
        description: "There was an error analyzing your text. Please try again.",
        variant: "destructive",
      })
    } finally {
      setIsLoading(false)
    }
  }

  const generateMockResult = (text: string): SentimentResult => {
    const lowerText = text.toLowerCase()

    // Simple sentiment detection based on keywords
    const positiveWords = [
      "love",
      "great",
      "amazing",
      "wonderful",
      "excellent",
      "fantastic",
      "good",
      "happy",
      "best",
      "phenomenal",
    ]
    const negativeWords = [
      "hate",
      "terrible",
      "awful",
      "bad",
      "worst",
      "horrible",
      "disappointed",
      "broken",
      "unhelpful",
    ]

    const positiveCount = positiveWords.filter((word) => lowerText.includes(word)).length
    const negativeCount = negativeWords.filter((word) => lowerText.includes(word)).length

    let label: "positive" | "negative" | "neutral"
    let score: number

    if (positiveCount > negativeCount) {
      label = "positive"
      score = Math.min(0.7 + positiveCount * 0.1, 0.95)
    } else if (negativeCount > positiveCount) {
      label = "negative"
      score = Math.min(0.7 + negativeCount * 0.1, 0.95)
    } else {
      label = "neutral"
      score = 0.6 + Math.random() * 0.2
    }

    // Generate mock tokens
    const words = text.split(/\s+/).slice(0, 8)
    const tokens = words
      .map((word) => ({
        term: word.toLowerCase().replace(/[^\w]/g, ""),
        weight: (Math.random() - 0.5) * 1.5,
      }))
      .filter((token) => token.term.length > 2)

    return { label, score, tokens }
  }

  const handleRetry = () => {
    if (analyzedText) {
      analyzeText(analyzedText)
    }
  }

  if (isLoading) {
    return (
      <div className="container mx-auto px-4 py-8">
        <div className="max-w-4xl mx-auto">
          <Card>
            <CardContent className="flex flex-col items-center justify-center py-12">
              <Loader2 className="h-8 w-8 animate-spin mb-4" />
              <h2 className="text-xl font-semibold mb-2">Analyzing Sentiment...</h2>
              <p className="text-muted-foreground text-center">
                Our AI is processing your text to determine its emotional tone
              </p>
            </CardContent>
          </Card>
        </div>
      </div>
    )
  }

  if (error) {
    return (
      <div className="container mx-auto px-4 py-8">
        <div className="max-w-4xl mx-auto">
          <Card>
            <CardContent className="flex flex-col items-center justify-center py-12">
              <AlertCircle className="h-8 w-8 text-destructive mb-4" />
              <h2 className="text-xl font-semibold mb-2">Analysis Failed</h2>
              <p className="text-muted-foreground text-center mb-4">{error}</p>
              <div className="flex gap-2">
                <Button onClick={handleRetry}>Try Again</Button>
                <Button variant="outline" onClick={() => router.push("/")}>
                  Go Back
                </Button>
              </div>
            </CardContent>
          </Card>
        </div>
      </div>
    )
  }

  return (
    <div className="container mx-auto px-4 py-8">
      <div className="max-w-6xl mx-auto">
        <div className="text-center mb-8">
          <h1 className="text-3xl font-bold mb-2">Sentiment Analysis Results</h1>
          <p className="text-muted-foreground">Here's what our AI detected in your text</p>
        </div>

        {/* Analyzed Text */}
        <Card className="mb-8">
          <CardContent className="p-6">
            <h3 className="font-semibold mb-2">Analyzed Text:</h3>
            <p className="text-muted-foreground italic">"{analyzedText}"</p>
          </CardContent>
        </Card>

        {result && (
          <div className="space-y-8">
            <ResultCards result={result} />
            <SentimentDashboard result={result} sessionHistory={sessionHistory} />
          </div>
        )}

        {/* Action Button */}
        <div className="text-center mt-8">
          <Button size="lg" onClick={() => router.push("/")}>
            Analyze Another Text
          </Button>
        </div>
      </div>
    </div>
  )
}
