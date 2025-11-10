"use client"

import { useEffect, useState } from "react"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import TrafficMap from "@/components/traffic-map"

import BackendStatus from "@/components/backend-status"
import LiveLocationPredictions from "@/components/live-location-predictions"
import TrafficHistoryChart from "@/components/traffic-history-chart"
import WeeklyTrafficHeatmap from "@/components/weekly-traffic-heatmap"
import CongestionDistribution from "@/components/congestion-distribution"

export default function TrafficPredictionPage() {
  const [userLocation, setUserLocation] = useState<{ lat: number; lng: number } | null>(null)
  const [locationError, setLocationError] = useState<string | null>(null)

  
  useEffect(() => {
    if (!navigator.geolocation) {
      setLocationError("Geolocation is not supported by this browser")
      return
    }

    const watchId = navigator.geolocation.watchPosition(
      (position) => {
        const { latitude, longitude } = position.coords
        setUserLocation({ lat: latitude, lng: longitude })
        setLocationError(null)
      },
      (error) => {
        console.error("Geolocation error:", error)
        setLocationError("Unable to get your location")
      },
      {
        enableHighAccuracy: true,
        timeout: 10000,
        maximumAge: 30000, 
      }
    )

    return () => navigator.geolocation.clearWatch(watchId)
  }, [])



  return (
    <main className="min-h-screen bg-background p-6">
      <BackendStatus />
      <div className="max-w-7xl mx-auto space-y-6">
        {}
        <div className="space-y-2">
          <h1 className="text-4xl font-bold text-foreground">Traffic Congestion Prediction</h1>
          <p className="text-muted-foreground">Real-time traffic analysis powered by AI-driven predictions.</p>
        </div>

        {}
        {locationError && (
          <Card className="border-yellow-500/50 bg-yellow-500/10">
            <CardContent className="pt-6">
              <p className="text-yellow-600 text-sm">⚠️ {locationError}</p>
            </CardContent>
          </Card>
        )}

        {userLocation && (
          <div className="border border-green-500/50 bg-gradient-to-r from-green-500/10 to-emerald-500/10 backdrop-blur rounded-lg px-4 py-3">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-3">
                <div className="relative">
                  <span className="text-2xl">📍</span>
                  <div className="absolute -top-0.5 -right-0.5 w-2 h-2 bg-green-500 rounded-full animate-pulse"></div>
                </div>
                <div className="flex items-center gap-3">
                  <span className="text-green-400 font-medium text-sm">Live location:</span>
                  <div className="flex items-center gap-2">
                    <span className="text-green-300 font-mono text-sm">
                      {userLocation.lat.toFixed(4)}, {userLocation.lng.toFixed(4)}
                    </span>
                  </div>
                </div>
              </div>
              <div className="flex items-center gap-2 bg-green-500/20 rounded-full px-3 py-1 border border-green-500/30">
                <div className="w-1.5 h-1.5 bg-green-400 rounded-full animate-pulse"></div>
                <span className="text-green-400 text-xs font-medium">Tracking</span>
              </div>
            </div>
          </div>
        )}

        {}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {}
          <Card className="border-border bg-card/50 backdrop-blur lg:col-span-2">
            <CardHeader>
              <CardTitle className="text-foreground">Traffic Map</CardTitle>
              <CardDescription>Real-time congestion levels across segments</CardDescription>
            </CardHeader>
            <CardContent>
              <TrafficMap
                segments={[]}
                selectedSegment={null}
                onSelectSegment={() => { }}
                userLocation={userLocation}
              />
            </CardContent>
          </Card>

          {}
          <CongestionDistribution />
        </div>

        {}
        {userLocation && (
          <LiveLocationPredictions userLocation={userLocation} />
        )}

        {}
        <TrafficHistoryChart />

        {}
        <WeeklyTrafficHeatmap />

      </div>
    </main>
  )
}