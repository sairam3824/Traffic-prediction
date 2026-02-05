"use client"

import { useEffect, useState } from "react"
import { GoogleMap, LoadScript, Polyline, Marker, InfoWindow, TrafficLayer, DirectionsRenderer, BicyclingLayer, TransitLayer } from "@react-google-maps/api"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { useTheme } from "next-themes"

interface RouteMapProps {
  origin: { lat: number; lng: number; name: string }
  destination: { lat: number; lng: number; name: string }
  polyline: string
  distance: number
  duration: number
  trafficDensity?: 'low' | 'medium' | 'high' | 'unknown'
  onTrafficLevelChange?: (level: 'low' | 'medium' | 'high' | 'unknown') => void
}

export default function RouteMap({ origin, destination, polyline, distance, duration, trafficDensity, onTrafficLevelChange }: RouteMapProps) {
  const [apiKey, setApiKey] = useState<string>("")
  const [selectedMarker, setSelectedMarker] = useState<"origin" | "destination" | null>(null)
  const [decodedPath, setDecodedPath] = useState<any[]>([])
  const [error, setError] = useState<string | null>(null)
  const [isLoading, setIsLoading] = useState(true)
  const [mapInstance, setMapInstance] = useState<google.maps.Map | null>(null)
  const [userHasInteracted, setUserHasInteracted] = useState(false)
  const [directions, setDirections] = useState<google.maps.DirectionsResult | null>(null)
  const [trafficSegments, setTrafficSegments] = useState<any[]>([])
  const [isLoadingTraffic, setIsLoadingTraffic] = useState(false)
  const [routeScores, setRouteScores] = useState<{ [key: number]: { score: number, avgTraffic: number, recommendation: string } }>({})
  const [bestRouteIndex, setBestRouteIndex] = useState<number>(0)
  const [showRouteOptimization, setShowRouteOptimization] = useState(true)
  const [currentRouteTrafficLevel, setCurrentRouteTrafficLevel] = useState<'low' | 'medium' | 'high' | 'unknown'>('unknown')

  const { theme } = useTheme()


  const isDark = theme === 'dark'
  const [showTraffic, setShowTraffic] = useState(true)
  const [showTransit, setShowTransit] = useState(false)
  const [showBicycling, setShowBicycling] = useState(false)
  const [travelMode, setTravelMode] = useState<'DRIVING' | 'TRANSIT' | 'WALKING' | 'BICYCLING'>('DRIVING')
  const [avoidTolls, setAvoidTolls] = useState(false)
  const [avoidHighways, setAvoidHighways] = useState(false)
  const [avoidFerries, setAvoidFerries] = useState(false)
  const [selectedRouteIndex, setSelectedRouteIndex] = useState(0)
  const [mapType, setMapType] = useState<'roadmap' | 'satellite' | 'hybrid' | 'terrain'>('roadmap')
  const [userLocation, setUserLocation] = useState<{ lat: number; lng: number } | null>(null)

  useEffect(() => {
    // Demo mode: Use environment variable directly
    const key = process.env.NEXT_PUBLIC_GOOGLE_MAPS_API_KEY
    if (key) {
      setApiKey(key)
      setIsLoading(false)
    } else {
      console.warn("Google Maps API key not found in environment variables")
      setIsLoading(false)
    }
  }, [])

  const analyzeAllRoutes = async (allRoutes: google.maps.DirectionsRoute[]) => {
    // Demo mode: Mock analysis
    const scores: { [key: number]: { score: number, avgTraffic: number, recommendation: string } } = {}

    // Simply pick the first one as best for demo
    scores[0] = {
      score: 85,
      avgTraffic: 45,
      recommendation: "✅ Excellent - Light traffic"
    }

    setRouteScores(scores)
    setBestRouteIndex(0)
    return 0
  }

  const getTrafficPredictionsForRoute = async (route: google.maps.DirectionsRoute) => {
    setIsLoadingTraffic(true)

    // Demo mode: Generate mock traffic segments
    setTimeout(() => {
      const segments: any[] = []
      const leg = route.legs[0]
      const steps = leg.steps

      for (let i = 0; i < steps.length; i++) {
        const step = steps[i]
        const trafficLevel = Math.random() > 0.7 ? 'high' : (Math.random() > 0.4 ? 'medium' : 'low')
        const prediction = trafficLevel === 'high' ? 75 : (trafficLevel === 'medium' ? 50 : 25)

        segments.push({
          stepIndex: i,
          trafficLevel,
          prediction,
          distance: step.distance?.value || 0,
          duration: step.duration?.value || 0,
          sampled: true
        })
      }

      setTrafficSegments(segments)
      calculateRouteTrafficLevel(segments)
      setIsLoadingTraffic(false)
    }, 800)
  }


  useEffect(() => {
    if (!mapInstance) return
    if (!origin || !destination) return

    try {
      const svc = new google.maps.DirectionsService()
      svc.route(
        {
          origin: { lat: origin.lat, lng: origin.lng },
          destination: { lat: destination.lat, lng: destination.lng },
          travelMode: travelMode as unknown as google.maps.TravelMode,
          provideRouteAlternatives: true,
          drivingOptions: {
            departureTime: new Date(),
            trafficModel: google.maps.TrafficModel.BEST_GUESS,
          },
          avoidFerries,
          avoidHighways,
          avoidTolls,
          region: "IN",
        },
        async (result, status) => {
          if (status === google.maps.DirectionsStatus.OK && result) {
            setDirections(result)


            if (result.routes.length > 1) {
              const bestIndex = await analyzeAllRoutes(result.routes)
              setSelectedRouteIndex(bestIndex)

              getTrafficPredictionsForRoute(result.routes[bestIndex])
            } else {
              setSelectedRouteIndex(0)

              getTrafficPredictionsForRoute(result.routes[0])
            }

            try {
              const bounds = result.routes?.[selectedRouteIndex]?.bounds
              if (bounds) mapInstance.fitBounds(bounds)
            } catch { }
          } else {
            console.warn("[v0] DirectionsService status:", status)
            setDirections(null)
            setTrafficSegments([])
            setRouteScores({})
          }
        },
      )
    } catch (e) {
      console.error("[v0] Directions error:", e)
      setDirections(null)
      setTrafficSegments([])
    }
  }, [origin, destination, mapInstance, travelMode, avoidFerries, avoidHighways, avoidTolls])

  const getRouteSummary = (route: google.maps.DirectionsRoute) => {
    const distanceMeters = route.legs.reduce((sum, l) => sum + (l.distance?.value || 0), 0)
    const durationSeconds = route.legs.reduce((sum, l) => sum + (l.duration?.value || 0), 0)
    return {
      distanceKm: distanceMeters / 1000,
      durationMin: Math.round(durationSeconds / 60),
    }
  }

  const getPolylineColor = () => {
    switch (trafficDensity) {
      case 'high':
        return '#ef4444'
      case 'medium':
        return '#f97316'
      case 'low':
        return '#10b981'
      default:
        return '#3b82f6'
    }
  }

  const getTrafficLabel = () => {
    switch (currentRouteTrafficLevel) {
      case 'high':
        return 'Heavy Traffic'
      case 'medium':
        return 'Moderate Traffic'
      case 'low':
        return 'Light Traffic'
      default:
        return 'Analyzing Traffic...'
    }
  }

  const getPolylineColorDynamic = () => {
    switch (currentRouteTrafficLevel) {
      case 'high':
        return '#ef4444'
      case 'medium':
        return '#f97316'
      case 'low':
        return '#10b981'
      default:
        return '#6b7280'
    }
  }


  const calculateRouteTrafficLevel = (segments: any[]) => {
    if (!segments || segments.length === 0) {
      setCurrentRouteTrafficLevel('unknown')
      onTrafficLevelChange?.('unknown')
      return
    }


    const totalPrediction = segments.reduce((sum, segment) => sum + (segment.prediction || 45), 0)
    const avgPrediction = totalPrediction / segments.length


    let overallLevel: 'low' | 'medium' | 'high'
    if (avgPrediction >= 65) {
      overallLevel = 'high'
    } else if (avgPrediction >= 40) {
      overallLevel = 'medium'
    } else {
      overallLevel = 'low'
    }

    setCurrentRouteTrafficLevel(overallLevel)
    onTrafficLevelChange?.(overallLevel)
  }

  const resetMapView = () => {
    if (mapInstance) {
      mapInstance.setCenter(mapCenter)
      mapInstance.setZoom(12)
    }
  }

  const mapContainerStyle = {
    width: "100%",
    height: "400px",
    borderRadius: "0.5rem",
  }

  const mapCenter = origin && destination && !userHasInteracted ? {
    lat: (origin.lat + destination.lat) / 2,
    lng: (origin.lng + destination.lng) / 2,
  } : {
    lat: 16.5062,
    lng: 80.6480
  }

  const mapOptions = {
    zoom: 12,
    mapTypeId: mapType,
    zoomControl: true,
    mapTypeControl: true,
    streetViewControl: true,
    fullscreenControl: true,
  }

  const handleLoadScriptError = (error: Error) => {
    console.error("[v0] LoadScript error:", error)
    setError(
      "Google Maps API failed to load. Ensure: 1) API key is valid, 2) Maps JavaScript API is enabled, 3) Places API is enabled, 4) Directions API is enabled in Google Cloud Console.",
    )
  }

  if (error) {
    return (
      <Card className="border-border bg-card/50 backdrop-blur">
        <CardHeader>
          <CardTitle className="text-foreground">Route Map</CardTitle>
          <CardDescription>
            Distance: {distance.toFixed(1)} km | Duration: {duration} min
            {currentRouteTrafficLevel !== 'unknown' && (
              <span className="ml-2 text-xs px-2 py-1 rounded" style={{ backgroundColor: getPolylineColorDynamic(), color: 'white' }}>
                {getTrafficLabel()}
              </span>
            )}
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="h-96 bg-slate-800 rounded-lg flex items-center justify-center p-4">
            <div className="text-center">
              <p className="text-red-400 font-medium mb-2">{error}</p>
              <div className="text-slate-500 text-sm space-y-1">
                <p>Setup Instructions:</p>
                <ol className="text-left inline-block">
                  <li>
                    1. Go to{" "}
                    <a
                      href="https://console.cloud.google.com"
                      target="_blank"
                      className="text-blue-400 hover:underline"
                      rel="noreferrer"
                    >
                      Google Cloud Console
                    </a>
                  </li>
                  <li>2. Enable: Maps JavaScript API, Places API, Directions API</li>
                  <li>3. Create an API key in Credentials</li>
                  <li>4. Add to environment: GOOGLE_MAPS_API_KEY=your-key</li>
                </ol>
              </div>
            </div>
          </div>
        </CardContent>
      </Card>
    )
  }

  if (!apiKey) {
    return (
      <Card className="border-border bg-card/50 backdrop-blur">
        <CardHeader>
          <CardTitle className="text-foreground">Route Map</CardTitle>
          <CardDescription>
            Distance: {distance.toFixed(1)} km | Duration: {duration} min
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="h-96 bg-slate-900/50 rounded-lg flex flex-col items-center justify-center border border-dashed border-slate-700 p-8 text-center space-y-4">
            <div className="bg-slate-800 p-4 rounded-full">
              <span className="text-4xl">🗺️</span>
            </div>
            <div>
              <h3 className="text-lg font-medium text-white mb-1">Map Visualization Unavailable</h3>
              <p className="text-slate-400 max-w-md mx-auto">
                To see the interactive map, you need to add your Google Maps API Key to the environment variables.
              </p>
            </div>
            <div className="text-xs text-slate-500 bg-slate-950/50 px-4 py-2 rounded font-mono">
              NEXT_PUBLIC_GOOGLE_MAPS_API_KEY=your_key_here
            </div>
          </div>
        </CardContent>
      </Card>
    )
  }

  if (isLoading) {
    return (
      <Card className="border-border bg-card/50 backdrop-blur">
        <CardHeader>
          <CardTitle className="text-foreground">Route Map</CardTitle>
          <CardDescription>
            Distance: {distance.toFixed(1)} km | Duration: {duration} min
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="h-96 bg-slate-800 rounded-lg flex items-center justify-center">
            <div className="text-center">
              <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-primary mx-auto mb-2"></div>
              <p className="text-slate-400">Loading map...</p>
            </div>
          </div>
        </CardContent>
      </Card>
    )
  }

  return (
    <Card className="border-border bg-card/50 backdrop-blur">
      <CardHeader>
        <CardTitle className="text-foreground">Route Map</CardTitle>
        <CardDescription>
          Distance: {distance.toFixed(1)} km | Duration: {duration} min
          {currentRouteTrafficLevel !== 'unknown' && (
            <span className="ml-2 text-xs px-2 py-1 rounded" style={{ backgroundColor: getPolylineColorDynamic(), color: 'white' }}>
              {getTrafficLabel()}
            </span>
          )}
          {isLoadingTraffic && (
            <span className="ml-2 text-xs px-2 py-1 rounded bg-gray-500 text-white">
              Analyzing Traffic...
            </span>
          )}
        </CardDescription>
      </CardHeader>
      <CardContent>
        <LoadScript
          googleMapsApiKey={apiKey}
          libraries={["geometry"]}
          onLoad={() => console.log("[v0] Google Maps loaded successfully")}
          onError={handleLoadScriptError}
        >
          <GoogleMap
            mapContainerStyle={mapContainerStyle}
            center={mapCenter}
            zoom={12}
            options={mapOptions}
            onLoad={(map) => {
              console.log("[v0] Map initialized successfully")
              setMapInstance(map)
            }}
            onDragStart={() => setUserHasInteracted(true)}
            onZoomChanged={() => setUserHasInteracted(true)}
          >
            { }
            {showTraffic && <TrafficLayer />}

            { }
            {showTransit && <TransitLayer />}
            {showBicycling && <BicyclingLayer />}
            {userLocation && (
              <Marker
                position={userLocation}
                title="Your location"
                icon={{
                  path: google.maps.SymbolPath.CIRCLE,
                  scale: 6,
                  fillColor: "#4285F4",
                  fillOpacity: 1,
                  strokeColor: "white",
                  strokeWeight: 2,
                }}
              />
            )}

            { }
            {directions ? (
              <>
                { }
                <DirectionsRenderer
                  options={{
                    directions,
                    routeIndex: selectedRouteIndex,
                    suppressMarkers: true,
                    preserveViewport: false,
                    polylineOptions: {
                      strokeColor: '#2563eb',
                      strokeOpacity: 0.1,
                      strokeWeight: 8,
                    },
                  }}
                />

                { }
                {directions.routes[selectedRouteIndex] && (() => {
                  const route = directions.routes[selectedRouteIndex]
                  const leg = route.legs[0]


                  let completePath: google.maps.LatLng[] = []
                  if (route.overview_polyline && window.google?.maps?.geometry?.encoding) {
                    try {
                      const polylinePoints = typeof route.overview_polyline === 'string'
                        ? route.overview_polyline
                        : (route.overview_polyline as any).points
                      completePath = window.google.maps.geometry.encoding.decodePath(polylinePoints)
                    } catch (error) {
                      console.error('Error decoding route polyline:', error)
                    }
                  }

                  if (completePath.length === 0) {
                    return null
                  }


                  const segmentSize = Math.max(1, Math.floor(completePath.length / 20))
                  const coloredSegments = []

                  for (let i = 0; i < completePath.length - segmentSize; i += segmentSize) {
                    const segmentPath = completePath.slice(i, i + segmentSize + 1)


                    let trafficLevel = 'medium'
                    let prediction = 45


                    const segmentIndex = Math.floor((i / completePath.length) * trafficSegments.length)
                    if (trafficSegments[segmentIndex]) {
                      trafficLevel = trafficSegments[segmentIndex].trafficLevel
                      prediction = trafficSegments[segmentIndex].prediction
                    }

                    const getSegmentColor = (level: string) => {
                      switch (level) {
                        case 'high': return '#ef4444'
                        case 'medium': return '#f97316'
                        case 'low': return '#10b981'
                        default: return '#6b7280'
                      }
                    }

                    coloredSegments.push(
                      <Polyline
                        key={`route-segment-${i}`}
                        path={segmentPath}
                        options={{
                          strokeColor: getSegmentColor(trafficLevel),
                          strokeWeight: 6,
                          strokeOpacity: 0.9,
                          geodesic: false,
                          zIndex: 1000,
                        }}
                        onClick={() => {
                          const midPoint = segmentPath[Math.floor(segmentPath.length / 2)]
                          const infoWindow = new google.maps.InfoWindow({
                            content: `
                              <div style="padding: 10px; min-width: 160px;">
                                <strong style="color: ${getSegmentColor(trafficLevel)};">
                                  ${trafficLevel.toUpperCase()} TRAFFIC
                                </strong><br/>
                                <span>Congestion: ${prediction.toFixed(1)}%</span><br/>
                                <span style="font-size: 11px; color: #666;">
                                  Segment ${Math.floor(i / segmentSize) + 1} of ${Math.ceil(completePath.length / segmentSize)}
                                </span>
                              </div>
                            `,
                            position: midPoint
                          })
                          infoWindow.open(mapInstance)
                        }}
                      />
                    )
                  }

                  return coloredSegments
                })()}

                { }
                {isLoadingTraffic && (
                  <div className="absolute top-4 left-4 bg-white/95 dark:bg-black/95 px-3 py-2 rounded-lg shadow-lg text-sm flex items-center gap-2 z-[1001]">
                    <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-primary"></div>
                    <span>Analyzing traffic on route...</span>
                  </div>
                )}

                { }
                <Marker
                  position={{ lat: origin.lat, lng: origin.lng }}
                  title={origin.name}
                  onClick={() => setSelectedMarker("origin")}
                  icon={{
                    path: "M0,-28a28,28 0 0,1 56,0c0,28-28,68-28,68S0,0 0,-28z",
                    fillColor: "#10b981",
                    fillOpacity: 1,
                    strokeColor: "#fff",
                    strokeWeight: 2,
                    scale: 0.7,
                  }}
                />
                <Marker
                  position={{ lat: destination.lat, lng: destination.lng }}
                  title={destination.name}
                  onClick={() => setSelectedMarker("destination")}
                  icon={{
                    path: "M0,-28a28,28 0 0,1 56,0c0,28-28,68-28,68S0,0 0,-28z",
                    fillColor: "#ef4444",
                    fillOpacity: 1,
                    strokeColor: "#fff",
                    strokeWeight: 2,
                    scale: 0.7,
                  }}
                />
              </>
            ) : (
              <>
                { }
                <Marker
                  position={{ lat: origin.lat, lng: origin.lng }}
                  title={origin.name}
                  onClick={() => setSelectedMarker("origin")}
                  icon={{
                    path: "M0,-28a28,28 0 0,1 56,0c0,28-28,68-28,68S0,0 0,-28z",
                    fillColor: "#10b981",
                    fillOpacity: 1,
                    strokeColor: "#fff",
                    strokeWeight: 2,
                    scale: 0.5,
                  }}
                />
                <Marker
                  position={{ lat: destination.lat, lng: destination.lng }}
                  title={destination.name}
                  onClick={() => setSelectedMarker("destination")}
                  icon={{
                    path: "M0,-28a28,28 0 0,1 56,0c0,28-28,68-28,68S0,0 0,-28z",
                    fillColor: "#ef4444",
                    fillOpacity: 1,
                    strokeColor: "#fff",
                    strokeWeight: 2,
                    scale: 0.5,
                  }}
                />
                { }
                {decodedPath.length > 0 && !directions && (
                  <Polyline
                    path={decodedPath}
                    options={{
                      strokeColor: getPolylineColor(),
                      strokeWeight: 5,
                      strokeOpacity: 0.9,
                      geodesic: true,
                    }}
                  />
                )}
              </>
            )}



            { }
            {selectedMarker === "origin" && (
              <InfoWindow position={{ lat: origin.lat, lng: origin.lng }} onCloseClick={() => setSelectedMarker(null)}>
                <div className="bg-popover text-popover-foreground p-3 rounded text-sm border border-border">
                  <h3 className="font-semibold">Origin</h3>
                  <p className="text-xs text-muted-foreground">{origin.name}</p>
                </div>
              </InfoWindow>
            )}

            { }
            {selectedMarker === "destination" && (
              <InfoWindow
                position={{ lat: destination.lat, lng: destination.lng }}
                onCloseClick={() => setSelectedMarker(null)}
              >
                <div className="bg-popover text-popover-foreground p-3 rounded text-sm border border-border">
                  <h3 className="font-semibold">Destination</h3>
                  <p className="text-xs text-muted-foreground">{destination.name}</p>
                </div>
              </InfoWindow>
            )}
          </GoogleMap>
        </LoadScript>

        { }
        <div className="flex flex-wrap items-center gap-4 mb-4 text-sm">
          <label className="flex items-center gap-2 text-foreground">
            <input
              type="checkbox"
              checked={showTraffic}
              onChange={(e) => setShowTraffic(e.target.checked)}
              className="accent-primary"
            />
            Traffic
          </label>
          <label className="flex items-center gap-2 text-foreground">
            <input
              type="checkbox"
              checked={showTransit}
              onChange={(e) => setShowTransit(e.target.checked)}
              className="accent-primary"
            />
            Transit
          </label>
          <label className="flex items-center gap-2 text-foreground">
            <input
              type="checkbox"
              checked={showBicycling}
              onChange={(e) => setShowBicycling(e.target.checked)}
              className="accent-primary"
            />
            Bicycling
          </label>
          <div className="h-5 w-px bg-border" />
          <div className="flex items-center gap-2">
            <span className="text-muted-foreground">Mode:</span>
            <div className="inline-flex overflow-hidden rounded-md border">
              <button
                className={`px-2 py-1 ${travelMode === 'DRIVING' ? 'bg-accent text-accent-foreground' : ''}`}
                onClick={() => setTravelMode('DRIVING')}
              >Driving</button>
              <button
                className={`px-2 py-1 border-l ${travelMode === 'TRANSIT' ? 'bg-accent text-accent-foreground' : ''}`}
                onClick={() => setTravelMode('TRANSIT')}
              >Transit</button>
              <button
                className={`px-2 py-1 border-l ${travelMode === 'WALKING' ? 'bg-accent text-accent-foreground' : ''}`}
                onClick={() => setTravelMode('WALKING')}
              >Walking</button>
              <button
                className={`px-2 py-1 border-l ${travelMode === 'BICYCLING' ? 'bg-accent text-accent-foreground' : ''}`}
                onClick={() => setTravelMode('BICYCLING')}
              >Bicycle</button>
            </div>
          </div>
          <div className="h-5 w-px bg-border" />
          <label className="flex items-center gap-2 text-foreground">
            <input type="checkbox" className="accent-primary" checked={avoidTolls} onChange={(e) => setAvoidTolls(e.target.checked)} />
            Avoid tolls
          </label>
          <label className="flex items-center gap-2 text-foreground">
            <input type="checkbox" className="accent-primary" checked={avoidHighways} onChange={(e) => setAvoidHighways(e.target.checked)} />
            Avoid highways
          </label>
          <label className="flex items-center gap-2 text-foreground">
            <input type="checkbox" className="accent-primary" checked={avoidFerries} onChange={(e) => setAvoidFerries(e.target.checked)} />
            Avoid ferries
          </label>
          <button
            onClick={() => {

              setAvoidHighways(false)

              if (mapInstance && origin && destination) {

                setDirections(null)
              }
            }}
            className="px-2 py-1 text-xs bg-green-600 text-white rounded hover:bg-green-700"
            title="Optimize route to avoid traffic"
          >
            🚀 Optimize Route
          </button>
          <div className="h-5 w-px bg-border" />
          <div className="flex items-center gap-2">
            <span className="text-muted-foreground">Map</span>
            <div className="inline-flex overflow-hidden rounded-md border">
              <button
                className={`px-2 py-1 ${mapType === 'roadmap' ? 'bg-accent text-accent-foreground' : ''}`}
                onClick={() => setMapType('roadmap')}
              >Default</button>
              <button
                className={`px-2 py-1 border-l ${mapType === 'satellite' ? 'bg-accent text-accent-foreground' : ''}`}
                onClick={() => setMapType('satellite')}
              >Satellite</button>
              <button
                className={`px-2 py-1 border-l ${mapType === 'hybrid' ? 'bg-accent text-accent-foreground' : ''}`}
                onClick={() => setMapType('hybrid')}
              >Hybrid</button>
              <button
                className={`px-2 py-1 border-l ${mapType === 'terrain' ? 'bg-accent text-accent-foreground' : ''}`}
                onClick={() => setMapType('terrain')}
              >Terrain</button>
            </div>
          </div>
          <button
            className="ml-auto px-3 py-1 rounded-md border bg-card text-foreground hover:bg-accent hover:text-accent-foreground"
            onClick={() => {
              if (!navigator.geolocation) return
              navigator.geolocation.getCurrentPosition(
                (pos) => {
                  const { latitude, longitude } = pos.coords
                  const loc = { lat: latitude, lng: longitude }
                  setUserLocation(loc)
                  try {
                    mapInstance?.setCenter(loc)
                    mapInstance?.setZoom(14)
                  } catch { }
                },
                () => { },
                { enableHighAccuracy: true, timeout: 10000 },
              )
            }}
          >
            My location
          </button>
        </div>
        { }
        {directions && directions.routes?.length > 1 && showRouteOptimization && (
          <div className="mb-4 p-4 bg-muted/50 rounded-lg border">
            <div className="flex items-center justify-between mb-3">
              <h3 className="font-semibold text-sm flex items-center gap-2">
                <svg className="w-4 h-4 text-primary" fill="currentColor" viewBox="0 0 20 20">
                  <path d="M8 16.5a1.5 1.5 0 11-3 0 1.5 1.5 0 013 0zM15 16.5a1.5 1.5 0 11-3 0 1.5 1.5 0 013 0z" />
                  <path d="M3 4a1 1 0 00-1 1v10a1 1 0 001 1h1.05a2.5 2.5 0 014.9 0H10a1 1 0 001-1V5a1 1 0 00-1-1H3zM14 7a1 1 0 00-1 1v6.05A2.5 2.5 0 0115.95 16H17a1 1 0 001-1V8a1 1 0 00-.293-.707L15 4.586A1 1 0 0014.414 4H14v3z" />
                </svg>
                Route Optimization
              </h3>
              <button
                onClick={() => setShowRouteOptimization(false)}
                className="text-xs text-muted-foreground hover:text-foreground"
              >
                ✕ Hide
              </button>
            </div>

            <div className="grid gap-2">
              {directions.routes.map((r, idx) => {
                const { distanceKm, durationMin } = getRouteSummary(r)
                const active = idx === selectedRouteIndex
                const isBest = idx === bestRouteIndex
                const score = routeScores[idx]

                return (
                  <button
                    key={idx}
                    className={`p-3 rounded-md border text-left text-sm transition-all ${active
                      ? 'bg-primary text-primary-foreground border-primary shadow-md'
                      : 'bg-card text-foreground border-border hover:bg-accent hover:text-accent-foreground'
                      }`}
                    onClick={() => {
                      setSelectedRouteIndex(idx)
                      getTrafficPredictionsForRoute(directions.routes[idx])


                      const routeScore = routeScores[idx]
                      if (routeScore) {
                        let routeLevel: 'low' | 'medium' | 'high'
                        if (routeScore.avgTraffic >= 65) {
                          routeLevel = 'high'
                        } else if (routeScore.avgTraffic >= 40) {
                          routeLevel = 'medium'
                        } else {
                          routeLevel = 'low'
                        }
                        setCurrentRouteTrafficLevel(routeLevel)
                      }
                    }}
                  >
                    <div className="flex items-center justify-between">
                      <div className="flex items-center gap-2">
                        <span className="font-medium">
                          Route {idx + 1} {isBest && '⭐'}
                        </span>
                        {score && (
                          <span className="text-xs px-2 py-1 rounded"
                            style={{
                              backgroundColor: score.avgTraffic < 40 ? '#10b981' :
                                score.avgTraffic < 60 ? '#f97316' : '#ef4444',
                              color: 'white'
                            }}>
                            {score.avgTraffic.toFixed(0)}% traffic
                          </span>
                        )}
                      </div>
                      <div className="text-xs text-right">
                        {durationMin} min · {distanceKm.toFixed(1)} km
                      </div>
                    </div>
                    {score && (
                      <div className="text-xs mt-1 opacity-90">
                        {score.recommendation}
                      </div>
                    )}
                  </button>
                )
              })}
            </div>

            { }
            <div className="mt-3 space-y-2">
              {routeScores[bestRouteIndex] && bestRouteIndex !== selectedRouteIndex && (
                <div className="p-2 bg-green-100 dark:bg-green-900/20 rounded text-xs">
                  💡 <strong>Better Route:</strong> Route {bestRouteIndex + 1} has {(routeScores[selectedRouteIndex]?.avgTraffic - routeScores[bestRouteIndex].avgTraffic).toFixed(0)}% less traffic
                </div>
              )}

              {routeScores[selectedRouteIndex]?.avgTraffic > 60 && (
                <div className="p-2 bg-yellow-100 dark:bg-yellow-900/20 rounded text-xs">
                  ⏰ <strong>Time Suggestion:</strong> Consider traveling 30-60 minutes earlier or later to avoid peak traffic
                </div>
              )}

              {routeScores[selectedRouteIndex]?.avgTraffic > 75 && (
                <div className="p-2 bg-red-100 dark:bg-red-900/20 rounded text-xs">
                  🚨 <strong>High Traffic Alert:</strong> Current route has severe congestion. Consider public transit or postponing travel.
                </div>
              )}
            </div>
          </div>
        )}

        { }
        {directions && directions.routes?.length > 1 && !showRouteOptimization && (
          <div className="mb-2 flex flex-wrap gap-2 items-center">
            <button
              onClick={() => setShowRouteOptimization(true)}
              className="text-xs px-2 py-1 bg-primary text-primary-foreground rounded hover:bg-primary/90"
            >
              <svg className="w-4 h-4 mr-1 inline" fill="currentColor" viewBox="0 0 20 20">
                <path d="M8 16.5a1.5 1.5 0 11-3 0 1.5 1.5 0 013 0zM15 16.5a1.5 1.5 0 11-3 0 1.5 1.5 0 013 0z" />
                <path d="M3 4a1 1 0 00-1 1v10a1 1 0 001 1h1.05a2.5 2.5 0 014.9 0H10a1 1 0 001-1V5a1 1 0 00-1-1H3zM14 7a1 1 0 00-1 1v6.05A2.5 2.5 0 0115.95 16H17a1 1 0 001-1V8a1 1 0 00-.293-.707L15 4.586A1 1 0 0014.414 4H14v3z" />
              </svg>
              Show Route Options
            </button>
            {directions.routes.map((r, idx) => {
              const { distanceKm, durationMin } = getRouteSummary(r)
              const active = idx === selectedRouteIndex
              const isBest = idx === bestRouteIndex
              return (
                <button
                  key={idx}
                  className={`px-3 py-1 rounded-md border text-sm ${active ? 'bg-primary text-primary-foreground border-primary' : 'bg-card text-foreground border-border hover:bg-accent hover:text-accent-foreground'}`}
                  onClick={() => {
                    setSelectedRouteIndex(idx)
                    getTrafficPredictionsForRoute(directions.routes[idx])


                    const routeScore = routeScores[idx]
                    if (routeScore) {
                      let routeLevel: 'low' | 'medium' | 'high'
                      if (routeScore.avgTraffic >= 65) {
                        routeLevel = 'high'
                      } else if (routeScore.avgTraffic >= 40) {
                        routeLevel = 'medium'
                      } else {
                        routeLevel = 'low'
                      }
                      setCurrentRouteTrafficLevel(routeLevel)
                    }
                  }}
                >
                  Route {idx + 1} {isBest && '⭐'} · {durationMin} min
                </button>
              )
            })}
          </div>
        )}
        { }
        <div className="mt-4 flex items-center justify-between gap-4 text-sm">
          <div className="flex gap-4 items-center">
            <div className="flex items-center gap-2">
              <div className="flex items-center justify-center w-6 h-4 rounded bg-green-500/20 border border-green-500/30">
                <svg className="w-3 h-3 text-green-600" fill="currentColor" viewBox="0 0 20 20">
                  <path fillRule="evenodd" d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z" clipRule="evenodd" />
                </svg>
              </div>
              <span className="text-foreground">Light Traffic</span>
            </div>
            <div className="flex items-center gap-2">
              <div className="flex items-center justify-center w-6 h-4 rounded bg-orange-500/20 border border-orange-500/30">
                <svg className="w-3 h-3 text-orange-600" fill="currentColor" viewBox="0 0 20 20">
                  <path fillRule="evenodd" d="M8.257 3.099c.765-1.36 2.722-1.36 3.486 0l5.58 9.92c.75 1.334-.213 2.98-1.742 2.98H4.42c-1.53 0-2.493-1.646-1.743-2.98l5.58-9.92zM11 13a1 1 0 11-2 0 1 1 0 012 0zm-1-8a1 1 0 00-1 1v3a1 1 0 002 0V6a1 1 0 00-1-1z" clipRule="evenodd" />
                </svg>
              </div>
              <span className="text-foreground">Moderate</span>
            </div>
            <div className="flex items-center gap-2">
              <div className="flex items-center justify-center w-6 h-4 rounded bg-red-500/20 border border-red-500/30">
                <svg className="w-3 h-3 text-red-600" fill="currentColor" viewBox="0 0 20 20">
                  <path fillRule="evenodd" d="M18 10a8 8 0 11-16 0 8 8 0 0116 0zm-7 4a1 1 0 11-2 0 1 1 0 012 0zm-1-9a1 1 0 00-1 1v4a1 1 0 102 0V6a1 1 0 00-1-1z" clipRule="evenodd" />
                </svg>
              </div>
              <span className="text-foreground">Heavy</span>
            </div>
            {isLoadingTraffic && (
              <div className="flex items-center gap-2 text-muted-foreground">
                <div className="animate-spin rounded-full h-3 w-3 border-b-2 border-primary"></div>
                <span className="text-xs">Loading traffic predictions...</span>
              </div>
            )}
            {trafficSegments.length > 0 && !isLoadingTraffic && (
              <div className="text-xs text-muted-foreground">
                ✅ {trafficSegments.length} segments
              </div>
            )}
            { }
            {routeScores[selectedRouteIndex] && routeScores[selectedRouteIndex].avgTraffic > 65 && (
              <div className="flex items-center gap-2 text-xs text-red-600 dark:text-red-400">
                ⚠️ High traffic detected - consider alternative route or time
              </div>
            )}
          </div>
          <button
            onClick={resetMapView}
            className="flex items-center gap-1 px-3 py-1 text-xs rounded-md bg-secondary hover:bg-secondary/80 text-secondary-foreground transition-colors"
            title="Reset map view"
          >
            🏠 Reset View
          </button>
        </div>
      </CardContent>
    </Card >
  )
}
