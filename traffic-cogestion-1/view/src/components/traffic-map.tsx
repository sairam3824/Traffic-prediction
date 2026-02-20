"use client"

import { useEffect, useState } from "react"
import { GoogleMap, LoadScript, TrafficLayer, TransitLayer, BicyclingLayer } from "@react-google-maps/api"


interface TrafficSegment {
  id: string
  segment_name: string
  latitude: number
  longitude: number
  road_type: string
}

interface TrafficMapProps {
  segments: TrafficSegment[]
  selectedSegment: string | null
  onSelectSegment: (id: string) => void
  userLocation?: { lat: number; lng: number } | null
}

export default function TrafficMap({ segments, selectedSegment, onSelectSegment, userLocation: propUserLocation }: TrafficMapProps) {

  const [mapCenter, setMapCenter] = useState({ lat: 16.5062, lng: 80.648 })
  const [apiKey, setApiKey] = useState<string>("")
  const [showTraffic, setShowTraffic] = useState(true)
  const [showTransit, setShowTransit] = useState(false)
  const [showBicycling, setShowBicycling] = useState(false)
  const [mapType, setMapType] = useState<'roadmap' | 'satellite' | 'hybrid' | 'terrain'>('roadmap')
  const [userLocation, setUserLocation] = useState<{ lat: number; lng: number } | null>(propUserLocation || null)

  useEffect(() => {
    // Demo mode: Use environment variable directly
    const key = process.env.NEXT_PUBLIC_GOOGLE_MAPS_API_KEY
    if (key) {
      setApiKey(key)
    } else {
      console.warn("Google Maps API key not found in environment variables")
    }
  }, [])



  useEffect(() => {
    if (propUserLocation) {
      setUserLocation(propUserLocation)
      setMapCenter(propUserLocation)
    } else if (segments.length > 0) {
      const avgLat = segments.reduce((sum, s) => sum + s.latitude, 0) / segments.length
      const avgLng = segments.reduce((sum, s) => sum + s.longitude, 0) / segments.length
      setMapCenter({ lat: avgLat, lng: avgLng })
    }
  }, [segments, propUserLocation])



  const mapContainerStyle = {
    width: "100%",
    height: "100%",
    borderRadius: "0.5rem",
  }

  const mapOptions = {
    zoom: 12,
    mapTypeId: mapType as unknown as google.maps.MapTypeId,
    zoomControl: true,
    mapTypeControl: true,
    streetViewControl: true,
    fullscreenControl: true,
  }

  if (!apiKey) {
    return (
      <div className="w-full h-96 bg-muted rounded-lg border border-border flex items-center justify-center">
        <p className="text-muted-foreground">Loading map...</p>
      </div>
    )
  }

  return (
    <div className="space-y-3 sm:space-y-4">
      <div className="h-[280px] sm:h-[400px] rounded-lg overflow-hidden">
      <LoadScript googleMapsApiKey={apiKey}>
        <GoogleMap mapContainerStyle={mapContainerStyle} center={mapCenter} zoom={12} options={mapOptions}>
          {showTraffic && <TrafficLayer />}
          {showTransit && <TransitLayer />}
          {showBicycling && <BicyclingLayer />}
        </GoogleMap>
      </LoadScript>
      </div>

      <div className="flex flex-col sm:flex-row sm:flex-wrap sm:items-center gap-3 sm:gap-4 text-sm">
        <div className="flex items-center gap-3 sm:gap-4">
          <label className="flex items-center gap-1.5 text-foreground cursor-pointer">
            <input type="checkbox" checked={showTraffic} onChange={(e) => setShowTraffic(e.target.checked)} className="accent-primary" />
            Traffic
          </label>
          <label className="flex items-center gap-1.5 text-foreground cursor-pointer">
            <input type="checkbox" checked={showTransit} onChange={(e) => setShowTransit(e.target.checked)} className="accent-primary" />
            Transit
          </label>
          <label className="flex items-center gap-1.5 text-foreground cursor-pointer">
            <input type="checkbox" checked={showBicycling} onChange={(e) => setShowBicycling(e.target.checked)} className="accent-primary" />
            Bicycling
          </label>
        </div>
        <div className="hidden sm:block h-5 w-px bg-border" />
        <div className="flex items-center justify-between sm:justify-start gap-2 sm:flex-1">
          <div className="flex items-center gap-2">
            <span className="text-muted-foreground hidden sm:inline">Map</span>
            <div className="inline-flex overflow-hidden rounded-md border text-xs sm:text-sm">
              <button className={`px-2 py-1.5 ${mapType === 'roadmap' ? 'bg-accent text-accent-foreground' : ''}`} onClick={() => setMapType('roadmap')}>Default</button>
              <button className={`px-2 py-1.5 border-l ${mapType === 'satellite' ? 'bg-accent text-accent-foreground' : ''}`} onClick={() => setMapType('satellite')}>Satellite</button>
              <button className={`px-2 py-1.5 border-l ${mapType === 'hybrid' ? 'bg-accent text-accent-foreground' : ''}`} onClick={() => setMapType('hybrid')}>Hybrid</button>
              <button className={`px-2 py-1.5 border-l ${mapType === 'terrain' ? 'bg-accent text-accent-foreground' : ''}`} onClick={() => setMapType('terrain')}>Terrain</button>
            </div>
          </div>
          <button
            className="px-3 py-1.5 rounded-md border bg-card text-foreground hover:bg-accent hover:text-accent-foreground text-xs sm:text-sm shrink-0"
            onClick={() => {
              if (!navigator.geolocation) return
              navigator.geolocation.getCurrentPosition(
                (pos) => {
                  const { latitude, longitude } = pos.coords
                  const loc = { lat: latitude, lng: longitude }
                  setUserLocation(loc)
                  setMapCenter(loc)
                },
                () => { },
                { enableHighAccuracy: true, timeout: 10000 },
              )
            }}
          >
            My location
          </button>
        </div>
      </div>

      { }
      <div className="flex flex-wrap gap-3 sm:gap-4 text-xs sm:text-sm">
        <div className="flex items-center gap-2">
          <div className="flex items-center justify-center w-6 h-4 rounded bg-green-500/20 border border-green-500/30">
            <svg className="w-3 h-3 text-green-600" fill="currentColor" viewBox="0 0 20 20">
              <path fillRule="evenodd" d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z" clipRule="evenodd" />
            </svg>
          </div>
          <span className="text-foreground">Free</span>
        </div>
        <div className="flex items-center gap-2">
          <div className="flex items-center justify-center w-6 h-4 rounded bg-yellow-500/20 border border-yellow-500/30">
            <svg className="w-3 h-3 text-yellow-600" fill="currentColor" viewBox="0 0 20 20">
              <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm1-12a1 1 0 10-2 0v4a1 1 0 00.293.707l2.828 2.829a1 1 0 101.415-1.415L11 9.586V6z" clipRule="evenodd" />
            </svg>
          </div>
          <span className="text-foreground">Moderate</span>
        </div>
        <div className="flex items-center gap-2">
          <div className="flex items-center justify-center w-6 h-4 rounded bg-orange-500/20 border border-orange-500/30">
            <svg className="w-3 h-3 text-orange-600" fill="currentColor" viewBox="0 0 20 20">
              <path fillRule="evenodd" d="M8.257 3.099c.765-1.36 2.722-1.36 3.486 0l5.58 9.92c.75 1.334-.213 2.98-1.742 2.98H4.42c-1.53 0-2.493-1.646-1.743-2.98l5.58-9.92zM11 13a1 1 0 11-2 0 1 1 0 012 0zm-1-8a1 1 0 00-1 1v3a1 1 0 002 0V6a1 1 0 00-1-1z" clipRule="evenodd" />
            </svg>
          </div>
          <span className="text-foreground">Heavy</span>
        </div>
        <div className="flex items-center gap-2">
          <div className="flex items-center justify-center w-6 h-4 rounded bg-red-500/20 border border-red-500/30">
            <svg className="w-3 h-3 text-red-600" fill="currentColor" viewBox="0 0 20 20">
              <path fillRule="evenodd" d="M18 10a8 8 0 11-16 0 8 8 0 0116 0zm-7 4a1 1 0 11-2 0 1 1 0 012 0zm-1-9a1 1 0 00-1 1v4a1 1 0 102 0V6a1 1 0 00-1-1z" clipRule="evenodd" />
            </svg>
          </div>
          <span className="text-foreground">Severe</span>
        </div>
      </div>
    </div>
  )
}
