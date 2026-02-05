"use client"

import type React from "react"

import { useState, useRef, useCallback } from "react"
import { Search, MapPin } from "lucide-react"
import { Card, CardContent } from "@/components/ui/card"
import RouteSearchHistory from "@/components/route-search-history"

interface LocationPrediction {
  place_id: string
  description: string
  main_text: string
  secondary_text?: string
  lat: number
  lng: number
}

interface RouteSearchProps {
  onRouteSelect: (origin: LocationPrediction, destination: LocationPrediction) => void
}

const MOCK_LOCATIONS: LocationPrediction[] = [
  {
    place_id: "1",
    description: "Vijayawada, Andhra Pradesh",
    main_text: "Vijayawada",
    secondary_text: "Andhra Pradesh",
    lat: 16.5062,
    lng: 80.6480
  },
  {
    place_id: "2",
    description: "Guntur, Andhra Pradesh",
    main_text: "Guntur",
    secondary_text: "Andhra Pradesh",
    lat: 16.3067,
    lng: 80.4365
  },
  {
    place_id: "3",
    description: "Amaravati, Andhra Pradesh",
    main_text: "Amaravati",
    secondary_text: "Andhra Pradesh",
    lat: 16.5417,
    lng: 80.5158
  },
  {
    place_id: "4",
    description: "Tenali, Andhra Pradesh",
    main_text: "Tenali",
    secondary_text: "Andhra Pradesh",
    lat: 16.2430,
    lng: 80.6400
  },
  {
    place_id: "5",
    description: "Visakhapatnam, Andhra Pradesh",
    main_text: "Visakhapatnam",
    secondary_text: "Andhra Pradesh",
    lat: 17.6868,
    lng: 83.2185
  },
  {
    place_id: "6",
    description: "Hyderabad, Telangana",
    main_text: "Hyderabad",
    secondary_text: "Telangana",
    lat: 17.3850,
    lng: 78.4867
  },
  {
    place_id: "7",
    description: "Tirupati, Andhra Pradesh",
    main_text: "Tirupati",
    secondary_text: "Andhra Pradesh",
    lat: 13.6288,
    lng: 79.4192
  },
  {
    place_id: "8",
    description: "Bangalore, Karnataka",
    main_text: "Bangalore",
    secondary_text: "Karnataka",
    lat: 12.9716,
    lng: 77.5946
  },
  {
    place_id: "9",
    description: "Chennai, Tamil Nadu",
    main_text: "Chennai",
    secondary_text: "Tamil Nadu",
    lat: 13.0827,
    lng: 80.2707
  },
  {
    place_id: "10",
    description: "Mumbai, Maharashtra",
    main_text: "Mumbai",
    secondary_text: "Maharashtra",
    lat: 19.0760,
    lng: 72.8777
  },
  {
    place_id: "11",
    description: "Delhi, New Delhi",
    main_text: "Delhi",
    secondary_text: "New Delhi",
    lat: 28.6139,
    lng: 77.2090
  },
  {
    place_id: "12",
    description: "Kolkata, West Bengal",
    main_text: "Kolkata",
    secondary_text: "West Bengal",
    lat: 22.5726,
    lng: 88.3639
  },
  {
    place_id: "13",
    description: "Pune, Maharashtra",
    main_text: "Pune",
    secondary_text: "Maharashtra",
    lat: 18.5204,
    lng: 73.8567
  },
  {
    place_id: "14",
    description: "Ahmedabad, Gujarat",
    main_text: "Ahmedabad",
    secondary_text: "Gujarat",
    lat: 23.0225,
    lng: 72.5714
  }
]

let saveToHistoryRef: ((origin: LocationPrediction, destination: LocationPrediction) => void) | null = null

export default function RouteSearch({ onRouteSelect }: RouteSearchProps) {
  const [originQuery, setOriginQuery] = useState("")
  const [destinationQuery, setDestinationQuery] = useState("")
  const [originPredictions, setOriginPredictions] = useState<LocationPrediction[]>([])
  const [destinationPredictions, setDestinationPredictions] = useState<LocationPrediction[]>([])
  const [selectedOrigin, setSelectedOrigin] = useState<LocationPrediction | null>(null)
  const [selectedDestination, setSelectedDestination] = useState<LocationPrediction | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const originInputRef = useRef<HTMLInputElement>(null)
  const destinationInputRef = useRef<HTMLInputElement>(null)
  const originDebounceRef = useRef<NodeJS.Timeout | undefined>(undefined)
  const destinationDebounceRef = useRef<NodeJS.Timeout | undefined>(undefined)

  const searchLocations = useCallback(async (query: string, isOrigin: boolean) => {
    try {
      setError(null)
      // Only set loading if query is typically long, but for mock instant is fine.
      // We'll skip loading state for instant default results to make it feel snappier.
      if (query.length > 0) setLoading(true)

      // Simulate network delay ONLY if typing (searching)
      if (query.length > 0) {
        await new Promise(resolve => setTimeout(resolve, 300))
      }

      const results = MOCK_LOCATIONS.filter(p =>
        p.main_text.toLowerCase().includes(query.toLowerCase()) ||
        p.secondary_text?.toLowerCase().includes(query.toLowerCase())
      )

      if (isOrigin) {
        setOriginPredictions(results)
      } else {
        setDestinationPredictions(results)
      }

    } catch (error) {
      console.error("[v0] Error searching locations:", error)
      setError("Network error. Please check your connection.")
    } finally {
      setLoading(false)
    }
  }, [])

  const handleOriginChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const value = e.target.value
    setOriginQuery(value)
    clearTimeout(originDebounceRef.current)

    originDebounceRef.current = setTimeout(() => {
      searchLocations(value, true)
    }, 300)
  }

  const handleDestinationChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const value = e.target.value
    setDestinationQuery(value)
    clearTimeout(destinationDebounceRef.current)

    destinationDebounceRef.current = setTimeout(() => {
      searchLocations(value, false)
    }, 300)
  }

  const handlePlanRoute = () => {
    if (selectedOrigin && selectedDestination) {

      if (saveToHistoryRef) {
        saveToHistoryRef(selectedOrigin, selectedDestination)
      }
      onRouteSelect(selectedOrigin, selectedDestination)
    }
  }


  const handleHistorySelect = (origin: LocationPrediction, destination: LocationPrediction) => {
    setSelectedOrigin(origin)
    setSelectedDestination(destination)
    setOriginQuery(origin.description)
    setDestinationQuery(destination.description)

    onRouteSelect(origin, destination)
  }


  const registerHistorySave = (saveFunction: (origin: LocationPrediction, destination: LocationPrediction) => void) => {
    saveToHistoryRef = saveFunction
  }

  return (
    <div className="space-y-4">
      <Card className="border-border bg-card/50 backdrop-blur">
        <CardContent className="p-6 space-y-4">
          {/* Origin */}
          <div className="space-y-2">
            <label className="text-sm font-medium text-foreground">From</label>
            <div className="relative">
              <MapPin className="absolute left-3 top-3 h-5 w-5 text-muted-foreground" />
              <input
                ref={originInputRef}
                type="text"
                placeholder="Enter starting location..."
                value={originQuery}
                onChange={handleOriginChange}
                onFocus={() => {
                  if (originPredictions.length === 0) searchLocations(originQuery, true)
                }}
                className="w-full pl-10 pr-4 py-2.5 bg-background border border-input rounded-xl text-foreground placeholder-muted-foreground focus:outline-none focus:ring-2 focus:ring-ring focus:border-input shadow-sm"
              />
              {originPredictions.length > 0 && (
                <div className="absolute top-full left-0 right-0 mt-1 z-50 max-h-64 overflow-y-auto rounded-xl border border-border bg-popover/95 backdrop-blur-sm shadow-xl divide-y divide-border">
                  {originPredictions.map((prediction) => (
                    <button
                      key={prediction.place_id}
                      onClick={() => {
                        setSelectedOrigin(prediction)
                        setOriginQuery(prediction.description)
                        setOriginPredictions([])
                      }}
                      className="group flex w-full items-start gap-3 px-4 py-2.5 text-left text-popover-foreground hover:bg-accent/60 hover:text-accent-foreground transition-colors"
                      role="option"
                    >
                      <MapPin className="mt-[2px] h-4 w-4 text-muted-foreground group-hover:text-accent-foreground" />
                      <div className="min-w-0 flex-1">
                        <div className="truncate text-sm font-medium">{prediction.main_text}</div>
                        {prediction.secondary_text && (
                          <div className="truncate text-xs text-muted-foreground">{prediction.secondary_text}</div>
                        )}
                      </div>
                    </button>
                  ))}
                </div>
              )}
            </div>
          </div>

          {/* Destination */}
          <div className="space-y-2">
            <label className="text-sm font-medium text-foreground">To</label>
            <div className="relative">
              <MapPin className="absolute left-3 top-3 h-5 w-5 text-muted-foreground" />
              <input
                ref={destinationInputRef}
                type="text"
                placeholder="Enter destination..."
                value={destinationQuery}
                onChange={handleDestinationChange}
                onFocus={() => {
                  if (destinationPredictions.length === 0) searchLocations(destinationQuery, false)
                }}
                className="w-full pl-10 pr-4 py-2.5 bg-background border border-input rounded-xl text-foreground placeholder-muted-foreground focus:outline-none focus:ring-2 focus:ring-ring focus:border-input shadow-sm"
              />
              {destinationPredictions.length > 0 && (
                <div className="absolute top-full left-0 right-0 mt-1 z-50 max-h-64 overflow-y-auto rounded-xl border border-border bg-popover/95 backdrop-blur-sm shadow-xl divide-y divide-border">
                  {destinationPredictions.map((prediction) => (
                    <button
                      key={prediction.place_id}
                      onClick={() => {
                        setSelectedDestination(prediction)
                        setDestinationQuery(prediction.description)
                        setDestinationPredictions([])
                      }}
                      className="group flex w-full items-start gap-3 px-4 py-2.5 text-left text-popover-foreground hover:bg-accent/60 hover:text-accent-foreground transition-colors"
                      role="option"
                    >
                      <MapPin className="mt-[2px] h-4 w-4 text-muted-foreground group-hover:text-accent-foreground" />
                      <div className="min-w-0 flex-1">
                        <div className="truncate text-sm font-medium">{prediction.main_text}</div>
                        {prediction.secondary_text && (
                          <div className="truncate text-xs text-muted-foreground">{prediction.secondary_text}</div>
                        )}
                      </div>
                    </button>
                  ))}
                </div>
              )}
            </div>
          </div>

          {error && (
            <div className="rounded-lg border p-3 text-sm bg-red-500/10 border-red-500/30 text-red-600 dark:text-red-300">
              {error}
            </div>
          )}

          {/* Action Button */}
          <button
            onClick={handlePlanRoute}
            disabled={!selectedOrigin || !selectedDestination || loading}
            className="w-full bg-primary hover:bg-primary/90 disabled:bg-muted disabled:text-muted-foreground disabled:cursor-not-allowed text-primary-foreground font-medium py-2 rounded-lg transition-colors flex items-center justify-center gap-2"
          >
            <Search className="h-4 w-4" />
            {loading ? "Searching..." : "Plan Route"}
          </button>

          {/* Route Summary */}
          {selectedOrigin && selectedDestination && (
            <div className="bg-muted rounded-lg p-3 space-y-2 text-sm">
              <div className="flex items-center gap-2 text-foreground">
                <MapPin className="h-4 w-4 text-green-500" />
                <span>{selectedOrigin.main_text}</span>
              </div>
              <div className="flex items-center gap-2 text-foreground">
                <MapPin className="h-4 w-4 text-red-500" />
                <span>{selectedDestination.main_text}</span>
              </div>
            </div>
          )}
        </CardContent>
      </Card>

      {/* History */}
      <RouteSearchHistory
        onHistorySelect={handleHistorySelect}
        onRegisterSave={registerHistorySave}
      />
    </div>
  )
}
