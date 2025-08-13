"use client"

import { useState, useEffect } from "react"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { Badge } from "@/components/ui/badge"
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table"
import { LiveCharts } from "@/components/live-charts"
import { Loader2, AlertCircle, Twitter } from "lucide-react"
import { useToast } from "@/hooks/use-toast"
import { apiClient } from "@/lib/api-client"

interface TweetData {
  id: string
  text: string
  label: "positive" | "negative" | "neutral"
  score: number
  created_at: string
}

export default function LiveTweetsPage() {
  const [query, setQuery] = useState("")
  const [limit, setLimit] = useState("25")
  const [tweets, setTweets] = useState<TweetData[]>([])
  const [isLoading, setIsLoading] = useState(false)
  const [hasSearched, setHasSearched] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [twitterStatus, setTwitterStatus] = useState<{
    enabled: boolean
    available: boolean
    reason?: string
  } | null>(null)
  const { toast } = useToast()

  // Check Twitter status on component mount
  useEffect(() => {
    checkTwitterStatus()
  }, [])

  const checkTwitterStatus = async () => {
    try {
      const status = await apiClient.getTwitterStatus()
      setTwitterStatus(status)
    } catch (error) {
      console.error("Failed to check Twitter status:", error)
      setTwitterStatus({ enabled: false, available: false, reason: "Failed to check status" })
    }
  }

  const handleSearch = async () => {
    if (!query.trim()) {
      toast({
        title: "Invalid Query",
        description: "Please enter a search term or hashtag",
        variant: "destructive",
      })
      return
    }

    setIsLoading(true)
    setHasSearched(true)
    setError(null)

    try {
      const result = await apiClient.analyzeTweets(query, Number.parseInt(limit))
      setTweets(result.items as TweetData[])

      toast({
        title: "Analysis Complete",
        description: `Analyzed ${result.items.length} tweets for "${query}"`,
      })
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : "Failed to analyze tweets"
      setError(errorMessage)
      setTweets([])

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

  // Show Twitter status warning if not available
  if (twitterStatus && !twitterStatus.available) {
    return (
      <div className="container mx-auto px-4 py-8">
        <div className="max-w-4xl mx-auto">
          <div className="text-center mb-8">
            <h1 className="text-3xl font-bold mb-2">Live Tweet Analysis</h1>
            <p className="text-muted-foreground">Analyze sentiment in real-time Twitter streams</p>
          </div>

          <Card>
            <CardContent className="flex flex-col items-center justify-center py-12">
              <Twitter className="h-12 w-12 text-muted-foreground mb-4" />
              <h2 className="text-xl font-semibold mb-2">Twitter Integration Unavailable</h2>
              <p className="text-muted-foreground text-center mb-4">
                {twitterStatus.reason || "Twitter integration is not properly configured"}
              </p>
              <p className="text-sm text-muted-foreground text-center">
                To enable Twitter analysis, configure the TWITTER_BEARER_TOKEN in your environment settings.
              </p>
              <Button onClick={checkTwitterStatus} className="mt-4">
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
          <h1 className="text-3xl font-bold mb-2">Live Tweet Analysis</h1>
          <p className="text-muted-foreground">Analyze sentiment in real-time Twitter streams</p>
        </div>

        {/* Search Form */}
        <Card className="mb-8">
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Twitter className="h-5 w-5" />
              Search Tweets
            </CardTitle>
            <CardDescription>Enter a hashtag or search query to analyze recent tweets</CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="flex gap-4">
              <Input
                placeholder="Enter hashtag or search term (e.g., #AI, coffee, etc.)"
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
                  <SelectItem value="10">10 tweets</SelectItem>
                  <SelectItem value="25">25 tweets</SelectItem>
                  <SelectItem value="50">50 tweets</SelectItem>
                  <SelectItem value="100">100 tweets</SelectItem>
                </SelectContent>
              </Select>
            </div>
            <Button className="w-full" size="lg" onClick={handleSearch} disabled={isLoading || !query.trim()}>
              {isLoading ? (
                <>
                  <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                  Analyzing Stream...
                </>
              ) : (
                <>
                  <Twitter className="mr-2 h-4 w-4" />
                  Analyze Stream
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

            {tweets.length > 0 && (
              <>
                {/* Live Charts */}
                <div className="mb-8">
                  <LiveCharts tweets={tweets} isLive={false} />
                </div>

                {/* Tweets Table */}
                <Card>
                  <CardHeader>
                    <CardTitle>Recent Tweets</CardTitle>
                    <CardDescription>Sentiment analysis results for "{query}"</CardDescription>
                  </CardHeader>
                  <CardContent>
                    <Table>
                      <TableHeader>
                        <TableRow>
                          <TableHead>Tweet</TableHead>
                          <TableHead>Sentiment</TableHead>
                          <TableHead>Score</TableHead>
                          <TableHead>Time</TableHead>
                        </TableRow>
                      </TableHeader>
                      <TableBody>
                        {tweets.slice(0, 10).map((tweet) => (
                          <TableRow key={tweet.id}>
                            <TableCell className="max-w-md">
                              <p className="truncate">{tweet.text}</p>
                            </TableCell>
                            <TableCell>
                              <Badge variant={getLabelVariant(tweet.label)} className="capitalize">
                                {tweet.label}
                              </Badge>
                            </TableCell>
                            <TableCell>{(tweet.score * 100).toFixed(1)}%</TableCell>
                            <TableCell className="text-muted-foreground">
                              {new Date(tweet.created_at).toLocaleTimeString()}
                            </TableCell>
                          </TableRow>
                        ))}
                      </TableBody>
                    </Table>
                    {tweets.length > 10 && (
                      <div className="mt-4 text-center text-sm text-muted-foreground">
                        Showing 10 of {tweets.length} tweets
                      </div>
                    )}
                  </CardContent>
                </Card>
              </>
            )}

            {!isLoading && !error && tweets.length === 0 && (
              <Card>
                <CardContent className="flex flex-col items-center justify-center py-12">
                  <Twitter className="h-8 w-8 text-muted-foreground mb-4" />
                  <p className="text-muted-foreground">No tweets found for "{query}"</p>
                  <p className="text-sm text-muted-foreground mt-2">Try a different search term or hashtag</p>
                </CardContent>
              </Card>
            )}
          </>
        )}
      </div>
    </div>
  )
}
