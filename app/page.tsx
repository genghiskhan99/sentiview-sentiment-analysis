import { SentimentForm } from "@/components/sentiment-form"

export default function HomePage() {
  return (
    <div className="container mx-auto px-4 py-8">
      {/* Hero Section */}
      <div className="text-center mb-12">
        <h1 className="text-4xl font-bold tracking-tight mb-4">Real-time Sentiment Analysis</h1>
        <p className="text-xl text-muted-foreground mb-2">
          Analyze the emotional tone of any text with AI-powered insights and confidence scores.
        </p>
        <p className="text-sm text-muted-foreground">
          Get instant sentiment analysis with detailed visualizations and token-level explanations.
        </p>
      </div>

      {/* Main Analysis Form */}
      <div className="max-w-2xl mx-auto">
        <SentimentForm />
      </div>
    </div>
  )
}
