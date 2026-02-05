"use client"

import { useState, useEffect } from "react"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Clock, MapPin, X, Trash2 } from "lucide-react"

interface LocationPrediction {
  place_id: string
  description: string
  main_text: string
  secondary_text?: string
  lat: number
  lng: number
}

interface SearchHistoryItem {
  id: string
  origin: LocationPrediction
  destination: LocationPrediction
  timestamp: number
  searchCount: number
}

interface RouteSearchHistoryProps {
  onHistorySelect: (origin: LocationPrediction, destination: LocationPrediction) => void
  onRegisterSave?: (saveFunction: (origin: LocationPrediction, destination: LocationPrediction) => void) => void
}

export default function RouteSearchHistory({ onHistorySelect, onRegisterSave }: RouteSearchHistoryProps) {
  const [history, setHistory] = useState<SearchHistoryItem[]>([])
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    // Demo mode: Always use localStorage
    loadHistoryFromLocalStorage()
    setLoading(false)
  }, [])

  const loadHistoryFromLocalStorage = () => {
    try {
      const savedHistory = localStorage.getItem('route-search-history')
      if (savedHistory) {
        const parsed = JSON.parse(savedHistory)
        setHistory(parsed)
      } else {
        // Initialize mock history
        const sampleHistory: SearchHistoryItem[] = [
          {
            id: 'sample-1',
            origin: {
              place_id: 'ChIJlfcOXx8XTjoRLJJAgbJqTtI',
              description: 'Vijayawada, Andhra Pradesh, India',
              main_text: 'Vijayawada',
              secondary_text: 'Andhra Pradesh, India',
              lat: 16.5062,
              lng: 80.648
            },
            destination: {
              place_id: 'ChIJ-_tVJHYYTjoRcVJNRtBvGTI',
              description: 'Guntur, Andhra Pradesh, India',
              main_text: 'Guntur',
              secondary_text: 'Andhra Pradesh, India',
              lat: 16.3067,
              lng: 80.4365
            },
            timestamp: Date.now() - 15 * 60 * 1000,
            searchCount: 3
          }
        ]
        setHistory(sampleHistory)
        localStorage.setItem('route-search-history', JSON.stringify(sampleHistory))
      }
    } catch (error) {
      console.error('Error loading from localStorage:', error)
      setHistory([])
    }
  }

  const saveToHistory = async (origin: LocationPrediction, destination: LocationPrediction) => {
    // Demo mode: Local storage only
    saveToLocalStorage(origin, destination)
  }

  const saveToLocalStorage = (origin: LocationPrediction, destination: LocationPrediction) => {
    try {
      const searchKey = `${origin.place_id}-${destination.place_id}`

      setHistory(prevHistory => {
        const existingIndex = prevHistory.findIndex(item =>
          item.origin.place_id === origin.place_id &&
          item.destination.place_id === destination.place_id
        )

        let newHistory: SearchHistoryItem[]

        if (existingIndex >= 0) {
          const existingItem = prevHistory[existingIndex]
          newHistory = [
            {
              ...existingItem,
              timestamp: Date.now(),
              searchCount: existingItem.searchCount + 1
            },
            ...prevHistory.filter((_, index) => index !== existingIndex)
          ]
        } else {
          const newItem: SearchHistoryItem = {
            id: searchKey,
            origin,
            destination,
            timestamp: Date.now(),
            searchCount: 1
          }
          newHistory = [newItem, ...prevHistory]
        }

        newHistory = newHistory.slice(0, 10)

        try {
          localStorage.setItem('route-search-history', JSON.stringify(newHistory))
        } catch (storageError) {
          console.error('Error saving to localStorage:', storageError)
        }

        return newHistory
      })
    } catch (error) {
      console.error('Error in saveToLocalStorage:', error)
    }
  }

  const removeFromHistory = async (id: string) => {
    setHistory(prevHistory => {
      const newHistory = prevHistory.filter(item => item.id !== id)
      localStorage.setItem('route-search-history', JSON.stringify(newHistory))
      return newHistory
    })
  }

  const clearHistory = async () => {
    setHistory([])
    localStorage.removeItem('route-search-history')
  }

  const handleHistoryClick = (item: SearchHistoryItem) => {
    onHistorySelect(item.origin, item.destination)
    saveToHistory(item.origin, item.destination)
  }

  const formatTimestamp = (timestamp: number) => {
    const now = Date.now()
    const diff = now - timestamp
    const seconds = Math.floor(diff / 1000)
    const minutes = Math.floor(diff / (1000 * 60))
    const hours = Math.floor(diff / (1000 * 60 * 60))
    const days = Math.floor(diff / (1000 * 60 * 60 * 24))

    if (seconds < 30) return 'Just now'
    if (seconds < 60) return 'Less than a minute ago'
    if (minutes === 1) return '1 minute ago'
    if (minutes < 60) return `${minutes} minutes ago`
    if (hours === 1) return '1 hour ago'
    if (hours < 24) return `${hours} hours ago`
    if (days === 1) return 'Yesterday'
    if (days < 7) return `${days} days ago`
    if (days < 30) return `${Math.floor(days / 7)} week${Math.floor(days / 7) > 1 ? 's' : ''} ago`

    const date = new Date(timestamp)
    const today = new Date()

    if (date.getFullYear() === today.getFullYear()) {
      return date.toLocaleDateString('en-US', {
        month: 'short',
        day: 'numeric'
      })
    }

    return date.toLocaleDateString('en-US', {
      month: 'short',
      day: 'numeric',
      year: 'numeric'
    })
  }

  useEffect(() => {
    if (onRegisterSave) {
      onRegisterSave(saveToHistory)
    }
  }, [onRegisterSave])

  if (history.length === 0) {
    return (
      <Card className="border-border bg-card/50 backdrop-blur">
        <CardHeader className="pb-3">
          <CardTitle className="text-sm font-medium text-foreground flex items-center gap-2">
            <Clock className="w-4 h-4" />
            Recent Searches
          </CardTitle>
        </CardHeader>
        <CardContent>
          <p className="text-sm text-muted-foreground text-center py-4">
            No recent searches. Your route history will appear here.
          </p>
        </CardContent>
      </Card>
    )
  }

  return (
    <Card className="border-border bg-card/50 backdrop-blur">
      <CardHeader className="pb-3">
        <div className="flex items-center justify-between">
          <CardTitle className="text-sm font-medium text-foreground flex items-center gap-2">
            <Clock className="w-4 h-4" />
            Recent Searches
          </CardTitle>
          <button
            onClick={clearHistory}
            className="text-muted-foreground hover:text-foreground transition-colors p-1 rounded"
            title="Clear all history"
          >
            <Trash2 className="w-4 h-4" />
          </button>
        </div>
      </CardHeader>
      <CardContent className="space-y-2">
        {history.map((item) => (
          <div
            key={item.id}
            className="group relative bg-muted/50 rounded-lg p-3 hover:bg-muted transition-colors cursor-pointer"
            onClick={() => handleHistoryClick(item)}
          >
            <button
              onClick={(e) => {
                e.stopPropagation()
                removeFromHistory(item.id)
              }}
              className="absolute top-2 right-2 opacity-0 group-hover:opacity-100 transition-opacity text-muted-foreground hover:text-foreground p-1 rounded"
              title="Remove from history"
            >
              <X className="w-3 h-3" />
            </button>

            <div className="space-y-2 pr-6">
              { }
              <div className="flex items-center gap-2 text-sm">
                <MapPin className="w-3 h-3 text-green-500 flex-shrink-0" />
                <span className="text-foreground font-medium truncate">
                  {item.origin.main_text}
                </span>
              </div>

              { }
              <div className="flex items-center gap-2 text-sm">
                <MapPin className="w-3 h-3 text-red-500 flex-shrink-0" />
                <span className="text-foreground font-medium truncate">
                  {item.destination.main_text}
                </span>
              </div>

              { }
              <div className="flex items-center justify-between text-xs text-muted-foreground">
                <span>{formatTimestamp(item.timestamp)}</span>
                {item.searchCount > 1 && (
                  <span>{item.searchCount} times</span>
                )}
              </div>
            </div>
          </div>
        ))}
      </CardContent>
    </Card>
  )
}