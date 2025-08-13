import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Progress } from "@/components/ui/progress"

interface SentimentResult {
  label: "positive" | "negative" | "neutral"
  score: number
  tokens: Array<{
    term: string
    weight: number
  }>
}

interface ResultCardsProps {
  result: SentimentResult
}

export function ResultCards({ result }: ResultCardsProps) {
  const getLabelVariant = (label: string) => {
    switch (label) {
      case "positive":
        return "default"
      case "negative":
        return "destructive"
      case "neutral":
        return "secondary"
      default:
        return "secondary"
    }
  }

  const getLabelColor = (label: string) => {
    switch (label) {
      case "positive":
        return "text-green-600"
      case "negative":
        return "text-red-600"
      case "neutral":
        return "text-gray-600"
      default:
        return "text-gray-600"
    }
  }

  return (
    <div className="space-y-6">
      {/* Main Results */}
      <div className="grid gap-6 md:grid-cols-2">
        {/* Sentiment Label */}
        <Card>
          <CardHeader>
            <CardTitle>Sentiment</CardTitle>
            <CardDescription>Overall emotional tone</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="flex items-center justify-center">
              <Badge variant={getLabelVariant(result.label)} className="text-lg px-4 py-2 capitalize">
                {result.label}
              </Badge>
            </div>
          </CardContent>
        </Card>

        {/* Confidence Score */}
        <Card>
          <CardHeader>
            <CardTitle>Confidence Score</CardTitle>
            <CardDescription>How certain our model is</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="text-center">
              <div className={`text-3xl font-bold mb-2 ${getLabelColor(result.label)}`}>
                {(result.score * 100).toFixed(1)}%
              </div>
              <Progress value={result.score * 100} className="w-full" />
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Key Words */}
      <Card>
        <CardHeader>
          <CardTitle>Key Words</CardTitle>
          <CardDescription>Words that most influenced this prediction</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="grid gap-3 sm:grid-cols-2">
            {result.tokens.map((token, index) => (
              <div key={index} className="flex items-center justify-between p-3 bg-muted rounded-lg">
                <span className="font-medium">{token.term}</span>
                <div className="flex items-center gap-2">
                  <span className="text-sm text-muted-foreground">{token.weight.toFixed(2)}</span>
                  <div className={`w-2 h-2 rounded-full ${token.weight > 0 ? "bg-green-500" : "bg-red-500"}`} />
                </div>
              </div>
            ))}
          </div>
        </CardContent>
      </Card>
    </div>
  )
}
