"use client"

import { useState, useEffect } from "react"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { Badge } from "@/components/ui/badge"
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table"
import { LiveCharts } from "@/components/live-charts"
import { Loader2, AlertCircle, ShoppingBag } from "lucide-react"
import { useToast } from "@/hooks/use-toast"
import { apiClient } from "@/lib/api-client"

interface ReviewData {
  id: string
  text: string
  label: "positive" | "negative" | "neutral"
  score: number
  created_at: string
}

export default function LiveReviewsPage() {
  const [query, setQuery] = useState("")
  const [limit, setLimit] = useState("25")
  const [reviews, setReviews] = useState<ReviewData[]>([])
  const [isLoading, setIsLoading] = useState(false)
  const [hasSearched, setHasSearched] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [amazonStatus, setAmazonStatus] = useState<{
    enabled: boolean
    available: boolean
    reason?: string
  } | null>(null)
  const { toast } = useToast()

  // Check Amazon status on component mount
  useEffect(() => {
    checkAmazonStatus()
  }, [])

  const checkAmazonStatus = async () => {
    try {
      const status = await apiClient.getAmazonStatus()
      setAmazonStatus(status)
    } catch (error) {
      console.error("Failed to check Amazon status:", error)
      setAmazonStatus({ enabled: false, available: false, reason: "Failed to check status" })
    }
  }

  const handleSearch = async () => {
    if (!query.trim()) {
      toast({
        title: "Invalid Query",
        description: "Please enter a search term or product category",
        variant: "destructive",
      })
      return
    }

    setIsLoading(true)
    setHasSearched(true)
    setError(null)

    try {
      const result = await apiClient.analyzeReviews(query, Number.parseInt(limit))
      setReviews(result.items as ReviewData[])

      toast({
        title: "Analysis Complete",
        description: `Analyzed ${result.items.length} reviews for "${query}"`,
      })
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : "Failed to analyze reviews"
      setError(errorMessage)
      setReviews([])

      toast({
        title: "Analysis Failed",
        description: errorMessage,
        variant: "destructive",
      })
    } finally {
      setIsLoading(false)
    }
  }

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

  // Show Amazon status warning if not available
  if (amazonStatus && !amazonStatus.available) {
    return (
      <div className="container mx-auto px-4 py-8">
        <div className="max-w-4xl mx-auto">
          <div className="text-center mb-8">
            <h1 className="text-3xl font-bold mb-2">Live Amazon Review Analysis</h1>
            <p className="text-muted-foreground">Analyze sentiment in Amazon product reviews</p>
          </div>

          <Card>
            <CardContent className="flex flex-col items-center justify-center py-12">
              <ShoppingBag className="h-12 w-12 text-muted-foreground mb-4" />
              <h2 className="text-xl font-semibold mb-2">Amazon Integration Unavailable</h2>
              <p className="text-muted-foreground text-center mb-4">
                {amazonStatus.reason || "Amazon integration is not properly configured"}
              </p>
              <p className="text-sm text-muted-foreground text-center">
                Amazon reviews service is currently unavailable. Please try again later.
              </p>
              <Button onClick={checkAmazonStatus} className="mt-4">
                Check Status Again
              </Button>
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
          <h1 className="text-3xl font-bold mb-2">Live Amazon Review Analysis</h1>
          <p className="text-muted-foreground">Analyze sentiment in Amazon product reviews</p>
        </div>

        {/* Search Form */}
        <Card className="mb-8">
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <ShoppingBag className="h-5 w-5" />
              Search Reviews
            </CardTitle>
            <CardDescription>Enter a product name or category to analyze Amazon reviews</CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="flex gap-4">
              <Input
                placeholder="Enter product name or category (e.g., smartphone, headphones, etc.)"
                className="flex-1"
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                onKeyDown={(e) => e.key === "Enter" && !isLoading && handleSearch()}
                disabled={isLoading}
              />
              <Select value={limit} onValueChange={setLimit} disabled={isLoading}>
                <SelectTrigger className="w-32">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="10">10 reviews</SelectItem>
                  <SelectItem value="25">25 reviews</SelectItem>
                  <SelectItem value="50">50 reviews</SelectItem>
                  <SelectItem value="100">100 reviews</SelectItem>
                </SelectContent>
              </Select>
            </div>
            <Button className="w-full" size="lg" onClick={handleSearch} disabled={isLoading || !query.trim()}>
              {isLoading ? (
                <>
                  <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                  Analyzing Reviews...
                </>
              ) : (
                <>
                  <ShoppingBag className="mr-2 h-4 w-4" />
                  Analyze Reviews
                </>
              )}
            </Button>
          </CardContent>
        </Card>

        {/* Results */}
        {hasSearched && (
          <>
            {error && (
              <Card className="mb-8">
                <CardContent className="flex flex-col items-center justify-center py-12">
                  <AlertCircle className="h-8 w-8 text-destructive mb-4" />
                  <h2 className="text-xl font-semibold mb-2">Analysis Failed</h2>
                  <p className="text-muted-foreground text-center mb-4">{error}</p>
                  <Button onClick={handleSearch}>Try Again</Button>
                </CardContent>
              </Card>
            )}

            {reviews.length > 0 && (
              <>
                {/* Live Charts */}
                <div className="mb-8">
                  <LiveCharts reviews={reviews} isLive={false} />
                </div>

                {/* Reviews Table */}
                <Card>
                  <CardHeader>
                    <CardTitle>Recent Reviews</CardTitle>
                    <CardDescription>Sentiment analysis results for "{query}"</CardDescription>
                  </CardHeader>
                  <CardContent>
                    <Table>
                      <TableHeader>
                        <TableRow>
                          <TableHead>Review</TableHead>
                          <TableHead>Sentiment</TableHead>
                          <TableHead>Score</TableHead>
                          <TableHead>Time</TableHead>
                        </TableRow>
                      </TableHeader>
                      <TableBody>
                        {reviews.slice(0, 10).map((review) => (
                          <TableRow key={review.id}>
                            <TableCell className="max-w-md">
                              <p className="truncate">{review.text}</p>
                            </TableCell>
                            <TableCell>
                              <Badge variant={getLabelVariant(review.label)} className="capitalize">
                                {review.label}
                              </Badge>
                            </TableCell>
                            <TableCell>{(review.score * 100).toFixed(1)}%</TableCell>
                            <TableCell className="text-muted-foreground">
                              {new Date(review.created_at).toLocaleTimeString()}
                            </TableCell>
                          </TableRow>
                        ))}
                      </TableBody>
                    </Table>
                    {reviews.length > 10 && (
                      <div className="mt-4 text-center text-sm text-muted-foreground">
                        Showing 10 of {reviews.length} reviews
                      </div>
                    )}
                  </CardContent>
                </Card>
              </>
            )}

            {!isLoading && !error && reviews.length === 0 && (
              <Card>
                <CardContent className="flex flex-col items-center justify-center py-12">
                  <ShoppingBag className="h-8 w-8 text-muted-foreground mb-4" />
                  <p className="text-muted-foreground">No reviews found for "{query}"</p>
                  <p className="text-sm text-muted-foreground mt-2">Try a different product name or category</p>
                </CardContent>
              </Card>
            )}
          </>
        )}
      </div>
    </div>
  )
}
