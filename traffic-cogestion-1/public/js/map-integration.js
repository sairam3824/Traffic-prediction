

class MapIntegrationHelper {
  constructor() {
    this.map = null;
    this.listeners = [];
  }

  
  async initializeMap(elementId, apiKey) {
    try {
      
      if (!apiKey) {
        const response = await fetch('/api/config/maps');
        const result = await response.json();
        if (result.success && result.data) {
          apiKey = result.data.apiKey;
        }
      }

      
      this.map = new TrafficMap(elementId, {
        apiKey: apiKey,
        center: { lat: 16.5062, lng: 80.6480 }, 
        zoom: 12,
        onEvent: (eventName, data) => this.handleMapEvent(eventName, data)
      });

      return this.map;
    } catch (error) {
      console.error('Map initialization error:', error);
      throw error;
    }
  }

  
  async loadTrafficSegments() {
    try {
      const response = await fetch('/api/traffic/segments');
      const result = await response.json();
      
      if (result.success && result.data) {
        const segments = result.data;
        
        
        for (const segment of segments) {
          await this.map.addTrafficSegment(segment);
        }
        
        return segments;
      }
      
      return [];
    } catch (error) {
      console.error('Error loading traffic segments:', error);
      return [];
    }
  }

  
  async predictTraffic(lat, lng, timestamp = null) {
    return await this.map.getPrediction(lat, lng, timestamp);
  }

  
  async predictRoute(waypoints) {
    return await this.map.getRoutePredictions(waypoints);
  }

  
  handleMapEvent(eventName, data) {
    console.log(`Map event: ${eventName}`, data);
    
    
    window.dispatchEvent(new CustomEvent('trafficMapEvent', {
      detail: { eventName, data }
    }));
  }

  
  on(eventName, callback) {
    document.addEventListener(`trafficMap:${eventName}`, (e) => {
      callback(e.detail);
    });
  }

  
  async getModelInfo() {
    try {
      const response = await fetch('/api/ucs-model-info');
      const result = await response.json();
      
      if (result.success) {
        return result.data;
      }
      
      return null;
    } catch (error) {
      console.error('Error fetching model info:', error);
      return null;
    }
  }

  
  toggleTrafficLayer(visible) {
    if (this.map) {
      this.map.toggleTrafficLayer(visible);
    }
  }

  
  recenterMap(lat, lng, zoom) {
    if (this.map) {
      this.map.recenter(lat, lng, zoom);
    }
  }

  
  destroy() {
    if (this.map) {
      this.map.destroy();
      this.map = null;
    }
  }
}


window.mapIntegration = new MapIntegrationHelper();


if (typeof module !== 'undefined' && module.exports) {
  module.exports = MapIntegrationHelper;
}
