

class TrafficMap {
  constructor(mapElementId, config = {}) {
    this.mapElement = document.getElementById(mapElementId);
    this.map = null;
    this.markers = [];
    this.trafficLayer = null;
    this.selectedSegment = null;
    this.config = {
      center: config.center || { lat: 16.5062, lng: 80.6480 }, 
      zoom: config.zoom || 12,
      apiKey: config.apiKey || '',
      ...config
    };
    
    this.init();
  }

  
  async init() {
    try {
      
      if (!window.google || !window.google.maps) {
        await this.loadGoogleMapsAPI();
      }
      
      
      this.map = new google.maps.Map(this.mapElement, {
        center: this.config.center,
        zoom: this.config.zoom,
        mapTypeControl: true,
        mapTypeControlOptions: {
          style: google.maps.MapTypeControlStyle.HORIZONTAL_BAR,
          position: google.maps.ControlPosition.TOP_RIGHT,
        },
        zoomControl: true,
        zoomControlOptions: {
          position: google.maps.ControlPosition.RIGHT_CENTER
        },
        streetViewControl: true,
        fullscreenControl: true,
        styles: this.getMapStyles()
      });

      
      this.trafficLayer = new google.maps.TrafficLayer();
      this.trafficLayer.setMap(this.map);
      
      console.log('✅ Interactive map initialized successfully');
      this.emit('initialized', { map: this.map });
    } catch (error) {
      console.error('❌ Error initializing map:', error);
      this.emit('error', { error });
    }
  }

  
  loadGoogleMapsAPI() {
    return new Promise((resolve, reject) => {
      if (window.google && window.google.maps) {
        resolve();
        return;
      }

      const script = document.createElement('script');
      script.src = `https:
      script.async = true;
      script.defer = true;
      script.onload = () => resolve();
      script.onerror = () => reject(new Error('Failed to load Google Maps API'));
      document.head.appendChild(script);
    });
  }

  
  addMarker(lat, lng, options = {}) {
    const marker = new google.maps.Marker({
      position: { lat, lng },
      map: this.map,
      title: options.title || '',
      icon: options.icon || this.getMarkerIcon(options.trafficLevel || 'unknown'),
      ...options
    });

    
    if (options.onClick) {
      marker.addListener('click', () => options.onClick(marker));
    }

    this.markers.push(marker);
    return marker;
  }

  
  async addTrafficSegment(segment) {
    try {
      
      const prediction = await this.getPrediction(segment.latitude, segment.longitude);
      
      const marker = this.addMarker(segment.latitude, segment.longitude, {
        title: segment.segment_name || `Segment ${segment.id}`,
        trafficLevel: prediction.traffic_level,
        onClick: (marker) => this.onSegmentClick(segment, prediction)
      });

      marker.segmentData = { ...segment, prediction };
      return marker;
    } catch (error) {
      console.error('Error adding traffic segment:', error);
      return null;
    }
  }

  
  async getPrediction(lat, lng, timestamp = null) {
    try {
      const response = await fetch('/api/ucs-predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          latitude: lat,
          longitude: lng,
          timestamp: timestamp || new Date().toISOString()
        })
      });

      const result = await response.json();
      if (result.success) {
        return result.data;
      }
      throw new Error(result.error || 'Prediction failed');
    } catch (error) {
      console.error('Prediction error:', error);
      return { prediction: 0, traffic_level: 'unknown', confidence: 'low' };
    }
  }

  
  onSegmentClick(segment, prediction) {
    this.selectedSegment = segment;
    
    const infoWindow = new google.maps.InfoWindow({
      content: this.createInfoWindowContent(segment, prediction)
    });

    infoWindow.setPosition({ lat: segment.latitude, lng: segment.longitude });
    infoWindow.open(this.map);
    
    this.emit('segmentSelected', { segment, prediction });
  }

  
  createInfoWindowContent(segment, prediction) {
    const trafficColor = this.getTrafficColor(prediction.traffic_level);
    return `
      <div style="padding: 10px; min-width: 200px;">
        <h3 style="margin: 0 0 10px 0; font-size: 16px; color: #333;">${segment.segment_name || 'Traffic Segment'}</h3>
        <div style="display: flex; align-items: center; margin-bottom: 8px;">
          <span style="display: inline-block; width: 12px; height: 12px; border-radius: 50%; background-color: ${trafficColor}; margin-right: 8px;"></span>
          <span style="font-weight: bold;">${prediction.traffic_level.toUpperCase()} Traffic</span>
        </div>
        <div style="font-size: 14px; color: #666;">
          <p style="margin: 5px 0;"><strong>Occupancy:</strong> ${prediction.prediction.toFixed(1)}%</p>
          <p style="margin: 5px 0;"><strong>Confidence:</strong> ${prediction.confidence}</p>
          <p style="margin: 5px 0;"><strong>Road Type:</strong> ${segment.road_type || 'N/A'}</p>
        </div>
      </div>
    `;
  }

  
  drawRoute(polyline, options = {}) {
    const path = google.maps.geometry.encoding.decodePath(polyline);
    
    const route = new google.maps.Polyline({
      path: path,
      geodesic: true,
      strokeColor: options.color || '#2563eb',
      strokeOpacity: options.opacity || 0.8,
      strokeWeight: options.weight || 5
    });

    route.setMap(this.map);
    return route;
  }

  
  async getRoutePredictions(waypoints) {
    try {
      const response = await fetch('/api/ucs-predict-route', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ waypoints })
      });

      const result = await response.json();
      if (result.success) {
        return result.data;
      }
      throw new Error(result.error || 'Route prediction failed');
    } catch (error) {
      console.error('Route prediction error:', error);
      return null;
    }
  }

  
  toggleTrafficLayer(show = true) {
    if (this.trafficLayer) {
      this.trafficLayer.setMap(show ? this.map : null);
    }
  }

  
  clearMarkers() {
    this.markers.forEach(marker => marker.setMap(null));
    this.markers = [];
  }

  
  getMarkerIcon(trafficLevel) {
    const colors = {
      low: '#10b981',
      medium: '#f59e0b',
      high: '#ef4444',
      unknown: '#6b7280'
    };
    
    return {
      path: google.maps.SymbolPath.CIRCLE,
      scale: 8,
      fillColor: colors[trafficLevel] || colors.unknown,
      fillOpacity: 0.9,
      strokeColor: '#ffffff',
      strokeWeight: 2
    };
  }

  
  getTrafficColor(trafficLevel) {
    const colors = {
      low: '#10b981',
      medium: '#f59e0b',
      high: '#ef4444',
      unknown: '#6b7280'
    };
    return colors[trafficLevel] || colors.unknown;
  }

  
  getMapStyles() {
    return [
      {
        featureType: 'poi',
        elementType: 'labels',
        stylers: [{ visibility: 'off' }]
      }
    ];
  }

  
  emit(eventName, data) {
    if (this.config.onEvent) {
      this.config.onEvent(eventName, data);
    }
    
    const event = new CustomEvent(`trafficMap:${eventName}`, { detail: data });
    document.dispatchEvent(event);
  }

  
  recenter(lat, lng, zoom) {
    this.map.setCenter({ lat, lng });
    if (zoom) this.map.setZoom(zoom);
  }

  
  destroy() {
    this.clearMarkers();
    if (this.trafficLayer) {
      this.trafficLayer.setMap(null);
    }
    this.map = null;
  }
}


if (typeof module !== 'undefined' && module.exports) {
  module.exports = TrafficMap;
} else {
  window.TrafficMap = TrafficMap;
}
