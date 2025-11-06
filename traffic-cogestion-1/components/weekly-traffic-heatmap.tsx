"use client"

import { useEffect, useState } from "react"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Calendar, Clock, TrendingUp } from "lucide-react"

interface HeatmapCell {
  day: string
  dayName: string
  hour: number
  congestion: number
  level: 'low' | 'moderate' | 'heavy' | 'severe'
  isPrediction: boolean
}

export default function WeeklyTrafficHeatmap() {
  const [heatmapData, setHeatmapData] = useState<HeatmapCell[]>([])
  const [loading, setLoading] = useState(true)
  const [selectedCell, setSelectedCell] = useState<HeatmapCell | null>(null)
  const [lastUpdate, setLastUpdate] = useState<Date | null>(null)

  const fetchWeeklyTrafficData = async () => {
    try {
      const response = await fetch("/api/traffic/observations")
      const result = await response.json()

      if (result.success && result.data) {
        const now = new Date()
        const heatmap: HeatmapCell[] = []

        // Generate data for 7 days (today + next 6 days)
        for (let dayOffset = 0; dayOffset < 7; dayOffset++) {
          const targetDate = new Date(now)
          targetDate.setDate(now.getDate() + dayOffset)
          
          const dayName = targetDate.toLocaleDateString('en-US', { weekday: 'short' })
          const dateStr = targetDate.toISOString().split('T')[0]
          const isToday = dayOffset === 0
          const currentHour = now.getHours()

          // Generate data for each hour (0-23)
          for (let hour = 0; hour < 24; hour++) {
            const isPrediction = dayOffset > 0 || (isToday && hour > currentHour)
            
            let congestion = 30 // Default moderate traffic

            if (!isPrediction) {
              // Historical data - try to get from API or generate realistic data
              const hourData = result.data.filter((obs: any) => {
                const obsDate = new Date(obs.timestamp)
                return obsDate.getHours() === hour && 
                       obsDate.toDateString() === targetDate.toDateString()
              })

              if (hourData.length > 0) {
                congestion = hourData.reduce((sum: number, obs: any) => 
                  sum + obs.occupancy_percent, 0) / hourData.length
              } else {
                congestion = generateRealisticTraffic(hour, dayName, false)
              }
            } else {
              // Predicted data
              congestion = generateRealisticTraffic(hour, dayName, true)
            }

            let level: 'low' | 'moderate' | 'heavy' | 'severe'
            if (congestion >= 70) level = 'severe'
            else if (congestion >= 50) level = 'heavy'
            else if (congestion >= 30) level = 'moderate'
            else level = 'low'

            heatmap.push({
              day: dateStr,
              dayName,
              hour,
              congestion: Math.round(congestion),
              level,
              isPrediction
            })
          }
        }

        setHeatmapData(heatmap)
        setLastUpdate(new Date())
      }
    } catch (error) {
      console.error("Error fetching weekly traffic data:", error)
    } finally {
      setLoading(false)
    }
  }

  // Generate realistic traffic patterns based on time and day
  const generateRealisticTraffic = (hour: number, dayName: string, isPrediction: boolean): number => {
    const isWeekend = dayName === 'Sat' || dayName === 'Sun'
    let baseCongestion = 30

    // Weekend patterns
    if (isWeekend) {
      if (hour >= 10 && hour <= 14) {
        // Weekend midday shopping/leisure
        baseCongestion = 45 + Math.random() * 15
      } else if (hour >= 18 && hour <= 21) {
        // Weekend evening entertainment
        baseCongestion = 50 + Math.random() * 20
      } else if (hour >= 0 && hour <= 6) {
        // Late night/early morning
        baseCongestion = 10 + Math.random() * 10
      } else {
        baseCongestion = 30 + Math.random() * 15
      }
    } 
    // Weekday patterns
    else {
      if (hour >= 7 && hour <= 9) {
        // Morning rush hour
        baseCongestion = 65 + Math.random() * 25
      } else if (hour >= 17 && hour <= 19) {
        // Evening rush hour
        baseCongestion = 70 + Math.random() * 20
      } else if (hour >= 10 && hour <= 16) {
        // Business hours
        baseCongestion = 40 + Math.random() * 20
      } else if (hour >= 20 && hour <= 23) {
        // Evening
        baseCongestion = 30 + Math.random() * 15
      } else if (hour >= 0 && hour <= 6) {
        // Night
        baseCongestion = 10 + Math.random() * 10
      } else {
        baseCongestion = 35 + Math.random() * 15
      }
    }

    // Add some variation for predictions
    if (isPrediction) {
      baseCongestion += (Math.random() - 0.5) * 10
    }

    return Math.max(5, Math.min(95, baseCongestion))
  }

  useEffect(() => {
    fetchWeeklyTrafficData()
    
    // Update every 10 minutes
    const interval = setInterval(fetchWeeklyTrafficData, 10 * 60 * 1000)
    return () => clearInterval(interval)
  }, [])

  const getCellColor = (level: string) => {
    switch (level) {
      case 'severe':
        return 'bg-red-500'
      case 'heavy':
        return 'bg-orange-500'
      case 'moderate':
        return 'bg-yellow-500'
      case 'low':
        return 'bg-green-500'
      default:
        return 'bg-gray-500'
    }
  }

  const getCellOpacity = (congestion: number) => {
    // Scale opacity based on congestion level
    const opacity = Math.max(0.3, Math.min(1, congestion / 100))
    return opacity
  }

  if (loading) {
    return (
      <Card className="border-border bg-card/50 backdrop-blur">
        <CardHeader>
          <CardTitle className="text-foreground flex items-center gap-2">
            <Calendar className="w-5 h-5" />
            7-Day Traffic Heatmap
          </CardTitle>
          <CardDescription>Hourly congestion patterns for the week ahead</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="h-96 flex items-center justify-center">
            <p className="text-muted-foreground">Loading heatmap data...</p>
          </div>
        </CardContent>
      </Card>
    )
  }

  // Group data by day
  const dayGroups = heatmapData.reduce((acc, cell) => {
    if (!acc[cell.dayName]) {
      acc[cell.dayName] = []
    }
    acc[cell.dayName].push(cell)
    return acc
  }, {} as Record<string, HeatmapCell[]>)

  const days = Object.keys(dayGroups)
  const hours = Array.from({ length: 24 }, (_, i) => i)

  return (
    <Card className="border-border bg-card/50 backdrop-blur">
      <CardHeader>
        <CardTitle className="text-foreground flex items-center gap-2">
          <Calendar className="w-5 h-5" />
          7-Day Traffic Heatmap
        </CardTitle>
        <CardDescription>
          Hourly congestion patterns for the week ahead
        </CardDescription>
        {lastUpdate && (
          <div className="flex items-center gap-2 text-xs text-muted-foreground">
            <Clock className="w-3 h-3" />
            Last updated: {lastUpdate.toLocaleTimeString()}
          </div>
        )}
      </CardHeader>
      <CardContent>
        <div className="space-y-4">
          {/* Heatmap Grid */}
          <div className="overflow-x-auto">
            <div className="inline-block min-w-full">
              {/* Hour labels */}
              <div className="flex mb-2">
                <div className="w-20 flex-shrink-0"></div>
                <div className="flex gap-0.5 flex-1">
                  {hours.map((hour) => (
                    <div
                      key={hour}
                      className="flex-1 text-center text-[10px] text-muted-foreground min-w-[24px]"
                    >
                      {hour % 3 === 0 ? `${hour}h` : ''}
                    </div>
                  ))}
                </div>
              </div>

              {/* Day rows */}
              {days.map((dayName) => {
                const dayCells = dayGroups[dayName]
                const isToday = dayName === new Date().toLocaleDateString('en-US', { weekday: 'short' })
                
                return (
                  <div key={dayName} className="flex mb-0.5 items-center">
                    {/* Day label */}
                    <div className={`w-20 flex-shrink-0 text-sm font-medium ${isToday ? 'text-primary' : 'text-foreground'}`}>
                      {dayName}
                      {isToday && <span className="text-xs text-primary ml-1">•</span>}
                    </div>
                    
                    {/* Hour cells */}
                    <div className="flex gap-0.5 flex-1">
                      {dayCells.map((cell) => (
                        <div
                          key={`${cell.day}-${cell.hour}`}
                          className={`flex-1 h-9 rounded-sm cursor-pointer transition-all hover:scale-105 hover:z-10 hover:shadow-lg relative min-w-[24px] ${getCellColor(cell.level)}`}
                          style={{ opacity: getCellOpacity(cell.congestion) }}
                          onClick={() => setSelectedCell(cell)}
                          title={`${cell.dayName} ${cell.hour}:00 - ${cell.congestion}% congestion`}
                        >
                          {cell.isPrediction && (
                            <div className="absolute top-0.5 right-0.5 w-1.5 h-1.5">
                              <div className="w-full h-full bg-white/50 rounded-full"></div>
                            </div>
                          )}
                        </div>
                      ))}
                    </div>
                  </div>
                )
              })}
            </div>
          </div>

          {/* Selected Cell Info */}
          {selectedCell && (
            <div className="mt-4 p-3 bg-muted/30 rounded-lg border border-border">
              <div className="flex items-start justify-between">
                <div>
                  <h4 className="font-semibold text-foreground flex items-center gap-2 text-sm">
                    {selectedCell.dayName} at {selectedCell.hour}:00
                    {selectedCell.isPrediction && (
                      <span className="text-[10px] px-1.5 py-0.5 bg-blue-500/20 text-blue-400 rounded">
                        Predicted
                      </span>
                    )}
                  </h4>
                  <p className="text-xs text-muted-foreground mt-1">
                    Congestion Level: <span className="font-medium text-foreground">{selectedCell.congestion}%</span>
                  </p>
                  <p className="text-xs text-muted-foreground">
                    Traffic Status: <span className={`font-medium ${
                      selectedCell.level === 'severe' ? 'text-red-400' :
                      selectedCell.level === 'heavy' ? 'text-orange-400' :
                      selectedCell.level === 'moderate' ? 'text-yellow-400' :
                      'text-green-400'
                    }`}>
                      {selectedCell.level.charAt(0).toUpperCase() + selectedCell.level.slice(1)}
                    </span>
                  </p>
                </div>
                <button
                  onClick={() => setSelectedCell(null)}
                  className="text-muted-foreground hover:text-foreground text-lg leading-none"
                >
                  ✕
                </button>
              </div>
            </div>
          )}

          {/* Legend */}
          <div className="flex flex-wrap items-center justify-between gap-4 pt-3 border-t border-border/50">
            <div className="flex items-center gap-3 text-xs">
              <span className="text-muted-foreground font-medium">Traffic Level:</span>
              <div className="flex items-center gap-1.5">
                <div className="w-3 h-3 bg-green-500 rounded-sm"></div>
                <span className="text-muted-foreground">Low</span>
              </div>
              <div className="flex items-center gap-1.5">
                <div className="w-3 h-3 bg-yellow-500 rounded-sm"></div>
                <span className="text-muted-foreground">Moderate</span>
              </div>
              <div className="flex items-center gap-1.5">
                <div className="w-3 h-3 bg-orange-500 rounded-sm"></div>
                <span className="text-muted-foreground">Heavy</span>
              </div>
              <div className="flex items-center gap-1.5">
                <div className="w-3 h-3 bg-red-500 rounded-sm"></div>
                <span className="text-muted-foreground">Severe</span>
              </div>
            </div>
            <div className="flex items-center gap-2 text-[10px] text-muted-foreground">
              <div className="w-2 h-2 bg-white/50 rounded-full"></div>
              <span>= Predicted data</span>
            </div>
          </div>

          {/* Summary Stats */}
          <div className="grid grid-cols-3 gap-4 pt-3 border-t border-border/50">
            <div className="text-center">
              <p className="text-3xl font-bold text-foreground">
                {Math.round(heatmapData.reduce((sum, cell) => sum + cell.congestion, 0) / heatmapData.length)}%
              </p>
              <p className="text-[11px] text-muted-foreground mt-0.5">Avg Weekly Congestion</p>
            </div>
            <div className="text-center">
              <p className="text-3xl font-bold text-red-400">
                {heatmapData.filter(cell => cell.level === 'severe' || cell.level === 'heavy').length}
              </p>
              <p className="text-[11px] text-muted-foreground mt-0.5">Peak Hours This Week</p>
            </div>
            <div className="text-center">
              <p className="text-3xl font-bold text-green-400">
                {heatmapData.filter(cell => cell.level === 'low').length}
              </p>
              <p className="text-[11px] text-muted-foreground mt-0.5">Low Traffic Hours</p>
            </div>
          </div>
        </div>
      </CardContent>
    </Card>
  )
}
