interface SentimentResponse {
  label: "positive" | "negative" | "neutral"
  score: number
  tokens: Array<{
    term: string
    weight: number
  }>
}

interface ReviewAnalysisResponse {
  items: Array<{
    id: string
    text: string
    label: string
    score: number
    created_at: string
  }>
  summary: {
    pos: number
    neu: number
    neg: number
  }
}

interface AmazonStatus {
  enabled: boolean
  available: boolean
  authenticated?: boolean
  reason?: string
}

class ApiClient {
  private baseUrl: string

  constructor() {
    this.baseUrl = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000"
  }

  async analyzeSentiment(text: string): Promise<SentimentResponse> {
    const response = await fetch(`${this.baseUrl}/predict`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ text }),
    })

    if (!response.ok) {
      const errorData = await response.json().catch(() => ({}))
      throw new Error(errorData.detail || `HTTP error! status: ${response.status}`)
    }

    return response.json()
  }

  async analyzeReviews(query: string, limit = 25): Promise<ReviewAnalysisResponse> {
    const params = new URLSearchParams({
      query,
      limit: limit.toString(),
    })

    const response = await fetch(`${this.baseUrl}/reviews/analyze?${params}`)

    if (!response.ok) {
      const errorData = await response.json().catch(() => ({}))
      throw new Error(errorData.detail || `HTTP error! status: ${response.status}`)
    }

    return response.json()
  }

  async getAmazonStatus(): Promise<AmazonStatus> {
    const response = await fetch(`${this.baseUrl}/amazon/status`)

    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`)
    }

    return response.json()
  }

  async healthCheck(): Promise<{ status: string; model_loaded: boolean; amazon_enabled: boolean }> {
    const response = await fetch(`${this.baseUrl}/health`)

    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`)
    }

    return response.json()
  }

  async batchAnalyzeSentiment(texts: string[]): Promise<{
    results: Array<{
      index: number
      label: string
      score: number
      tokens: Array<{ term: string; weight: number }>
    }>
    count: number
    processing_time: number
  }> {
    const response = await fetch(`${this.baseUrl}/predict/batch`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify(texts),
    })

    if (!response.ok) {
      const errorData = await response.json().catch(() => ({}))
      throw new Error(errorData.detail || `HTTP error! status: ${response.status}`)
    }

    return response.json()
  }
}

export const apiClient = new ApiClient()
