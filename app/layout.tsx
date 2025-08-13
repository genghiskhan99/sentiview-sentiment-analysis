import type React from "react"
import type { Metadata } from "next"
import { GeistSans } from "geist/font/sans"
import { GeistMono } from "geist/font/mono"
import "./globals.css"
import { Toaster } from "@/components/ui/toaster"

export const metadata: Metadata = {
  title: "Sentiview - Real-time Sentiment Analysis",
  description: "Analyze sentiment in text and live tweets with AI-powered insights and visualizations",
  generator: "v0.app",
}

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode
}>) {
  return (
    <html lang="en">
      <head>
        <style>{`
html {
  font-family: ${GeistSans.style.fontFamily};
  --font-sans: ${GeistSans.variable};
  --font-mono: ${GeistMono.variable};
}
        `}</style>
      </head>
      <body>
        <header className="border-b bg-background/95 backdrop-blur supports-[backdrop-filter]:bg-background/60">
          <div className="container flex h-14 items-center">
            <div className="mr-4 flex">
              <a className="mr-6 flex items-center space-x-2" href="/">
                <span className="font-bold text-xl">Sentiview</span>
              </a>
              <nav className="flex items-center space-x-6 text-sm font-medium">
                <a href="/" className="transition-colors hover:text-foreground/80">
                  Home
                </a>
                <a href="/live" className="transition-colors hover:text-foreground/80">
                  Live Tweets
                </a>
              </nav>
            </div>
          </div>
        </header>
        <main className="flex-1">{children}</main>
        <footer className="border-t py-6 md:py-0">
          <div className="container flex flex-col items-center justify-between gap-4 md:h-24 md:flex-row">
            <div className="flex flex-col items-center gap-4 px-8 md:flex-row md:gap-2 md:px-0">
              <p className="text-center text-sm leading-loose text-muted-foreground md:text-left">
                Built with Next.js and FastAPI. No user text is stored.
              </p>
            </div>
            <div className="flex items-center space-x-4">
              <a href="#" className="text-sm text-muted-foreground hover:text-foreground">
                GitHub
              </a>
              <a href="#" className="text-sm text-muted-foreground hover:text-foreground">
                README
              </a>
            </div>
          </div>
        </footer>
        <Toaster />
      </body>
    </html>
  )
}
