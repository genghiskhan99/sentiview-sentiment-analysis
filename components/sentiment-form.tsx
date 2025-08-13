"use client"

import type React from "react"

import { useState } from "react"
import { useRouter } from "next/navigation"
import { z } from "zod"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Textarea } from "@/components/ui/textarea"
import { Badge } from "@/components/ui/badge"
import { Loader2 } from "lucide-react"
import { useToast } from "@/hooks/use-toast"

const sentimentSchema = z.object({
  text: z.string().min(1, "Please enter some text to analyze").max(5000, "Text must be less than 5000 characters"),
})

interface SentimentFormProps {
  onAnalyze?: (text: string) => void
}

export function SentimentForm({ onAnalyze }: SentimentFormProps) {
  const [text, setText] = useState("")
  const [isLoading, setIsLoading] = useState(false)
  const [errors, setErrors] = useState<string[]>([])
  const router = useRouter()
  const { toast } = useToast()

  const exampleTexts = [
    "I absolutely loved this movie! The acting was phenomenal and the plot kept me engaged throughout.",
    "The service was okay, nothing special but not terrible either. Average experience overall.",
    "This product is completely broken and the customer support is unhelpful. Very disappointed.",
  ]

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    setErrors([])

    try {
      const validatedData = sentimentSchema.parse({ text })
      setIsLoading(true)

      if (onAnalyze) {
        onAnalyze(validatedData.text)
      } else {
        // Store text in sessionStorage and navigate to results
        sessionStorage.setItem("analysisText", validatedData.text)
        router.push("/results")
      }
    } catch (error) {
      if (error instanceof z.ZodError) {
        setErrors(error.errors.map((err) => err.message))
      } else {
        toast({
          title: "Error",
          description: "Failed to analyze text. Please try again.",
          variant: "destructive",
        })
      }
    } finally {
      setIsLoading(false)
    }
  }

  const handleExampleClick = (exampleText: string) => {
    setText(exampleText)
  }

  return (
    <div className="space-y-6">
      <Card>
        <CardHeader>
          <CardTitle>Analyze Text Sentiment</CardTitle>
          <CardDescription>Enter any text to analyze its emotional sentiment</CardDescription>
        </CardHeader>
        <CardContent>
          <form onSubmit={handleSubmit} className="space-y-4">
            <div className="space-y-2">
              <Textarea
                placeholder="Enter your text here to analyze its sentiment..."
                className="min-h-32 resize-none"
                value={text}
                onChange={(e) => setText(e.target.value)}
                disabled={isLoading}
              />
              <div className="flex justify-between text-sm text-muted-foreground">
                <span>{text.length}/5000 characters</span>
                {errors.length > 0 && <span className="text-destructive">{errors[0]}</span>}
              </div>
            </div>
            <Button type="submit" className="w-full" size="lg" disabled={isLoading || !text.trim()}>
              {isLoading ? (
                <>
                  <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                  Analyzing...
                </>
              ) : (
                "Analyze Sentiment"
              )}
            </Button>
          </form>
        </CardContent>
      </Card>

      {/* Example Texts */}
      <div>
        <h3 className="text-lg font-semibold mb-4 text-center">Try these examples:</h3>
        <div className="grid gap-3">
          {exampleTexts.map((exampleText, index) => (
            <Card
              key={index}
              className="cursor-pointer hover:bg-accent/50 transition-colors"
              onClick={() => handleExampleClick(exampleText)}
            >
              <CardContent className="p-4">
                <div className="flex items-start justify-between gap-3">
                  <p className="text-sm flex-1">{exampleText}</p>
                  <Badge variant="outline" className="text-xs">
                    Try this
                  </Badge>
                </div>
              </CardContent>
            </Card>
          ))}
        </div>
      </div>
    </div>
  )
}
