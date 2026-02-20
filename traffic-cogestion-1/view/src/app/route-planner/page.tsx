"use client"

import { useState } from "react"
import RouteSearch from "@/components/route-search"
import RouteMap from "@/components/route-map"
import RouteDetails from "@/components/route-details"
import CongestionForecast from "@/components/congestion-forecast"
import { Card, CardContent } from "@/components/ui/card"

interface LocationPrediction {
  place_id: string
  description: string
  main_text: string
  secondary_text?: string
  lat: number
  lng: number
}

interface RouteStep {
  instruction: string
  distance_km: number
  duration_minutes: number
  traffic_density: 'low' | 'medium' | 'high' | 'unknown'
}

interface RouteData {
  id: string
  distance_km: number
  duration_minutes: number
  route_polyline: string
  traffic_density?: 'low' | 'medium' | 'high' | 'unknown'
  steps?: RouteStep[]
  congestion_level?: number
}

export default function RoutePlannerPage() {
  const [selectedRoute, setSelectedRoute] = useState<RouteData | null>(null)
  const [origin, setOrigin] = useState<LocationPrediction | null>(null)
  const [destination, setDestination] = useState<LocationPrediction | null>(null)
  const [loading, setLoading] = useState(false)
  const [dynamicTrafficLevel, setDynamicTrafficLevel] = useState<'low' | 'medium' | 'high' | 'unknown'>('unknown')

  const handleRouteSelect = async (originLocation: LocationPrediction, destinationLocation: LocationPrediction) => {
    // Demo mode: Mock route planning
    setLoading(true)
    setOrigin(originLocation)
    setDestination(destinationLocation)

    // Simulate network delay
    setTimeout(() => {
      const mockRoute: RouteData = {
        id: "mock_route_" + Date.now(),
        distance_km: 12.5,
        duration_minutes: 25,
        route_polyline: "", // We can leave this empty or provide a dummy encoded polyline if needed, but for now we'll handle it in the map component if possible or just use a straight line in the mock map logic
        traffic_density: "medium",
        congestion_level: 45,
        steps: [
          {
            instruction: "Head north on Main St",
            distance_km: 2.0,
            duration_minutes: 5,
            traffic_density: "low"
          },
          {
            instruction: "Turn right onto Highway 101",
            distance_km: 8.5,
            duration_minutes: 15,
            traffic_density: "high"
          },
          {
            instruction: "Take exit 42 towards Downtown",
            distance_km: 2.0,
            duration_minutes: 5,
            traffic_density: "medium"
          }
        ]
      }
      setSelectedRoute(mockRoute)
      setLoading(false)
    }, 1500)
  }

  return (
    <main className="min-h-screen bg-background p-4 sm:p-6">
      <div className="max-w-7xl mx-auto space-y-4 sm:space-y-6">
        { }
        <div className="space-y-1 sm:space-y-2">
          <h1 className="text-2xl sm:text-4xl font-bold text-foreground">Route Planner</h1>
          <p className="text-sm sm:text-base text-muted-foreground">Plan your route and check real-time congestion predictions</p>
        </div>

        { }
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          { }
          <div className="lg:col-span-1">
            <RouteSearch onRouteSelect={handleRouteSelect} />
          </div>

          { }
          <div className="lg:col-span-2 space-y-6">
            {selectedRoute && origin && destination ? (
              <>
                <RouteMap
                  origin={{ lat: origin.lat, lng: origin.lng, name: origin.main_text }}
                  destination={{ lat: destination.lat, lng: destination.lng, name: destination.main_text }}
                  polyline={selectedRoute.route_polyline}
                  distance={selectedRoute.distance_km}
                  duration={selectedRoute.duration_minutes}
                  trafficDensity={selectedRoute.traffic_density}
                  onTrafficLevelChange={setDynamicTrafficLevel}
                />
                <RouteDetails
                  distance={selectedRoute.distance_km}
                  duration={selectedRoute.duration_minutes}
                  trafficDensity={dynamicTrafficLevel !== 'unknown' ? dynamicTrafficLevel : selectedRoute.traffic_density}
                  steps={selectedRoute.steps || []}
                  congestionLevel={selectedRoute.congestion_level || 0}
                />
                <CongestionForecast routeId={selectedRoute.id} />
              </>
            ) : (
              <Card className="border-border bg-card/50 backdrop-blur">
                <CardContent className="p-6 sm:p-12 text-center">
                  <p className="text-sm sm:text-base text-muted-foreground">
                    {loading
                      ? "Planning route..."
                      : "Select origin and destination to view route and congestion forecast"}
                  </p>
                </CardContent>
              </Card>
            )}
          </div>
        </div>
      </div>
    </main>
  )
}
