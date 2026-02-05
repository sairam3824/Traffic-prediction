

export interface NotificationPayload {
  subject: string
  message: string
  alertType: "congestion" | "incident" | "weather"
  severity: "low" | "medium" | "high" | "critical"
  segmentId: string
  timestamp: Date
}

export class NotificationService {
  
  async sendPushNotification(payload: NotificationPayload): Promise<boolean> {
    try {
      
      console.log(`[PUSH] Sending notification:`, {
        subject: payload.subject,
        message: payload.message,
        severity: payload.severity,
        alertType: payload.alertType,
        segmentId: payload.segmentId,
        timestamp: payload.timestamp,
      })

      
      return true
    } catch (error) {
      console.error("Error sending push notification:", error)
      return false
    }
  }

  
  async sendInAppNotification(payload: NotificationPayload): Promise<boolean> {
    try {
      
      console.log(`[IN-APP] Storing notification:`, {
        subject: payload.subject,
        message: payload.message,
        severity: payload.severity,
        alertType: payload.alertType,
      })

      
      return true
    } catch (error) {
      console.error("Error storing in-app notification:", error)
      return false
    }
  }

  
  async sendNotification(payload: NotificationPayload): Promise<void> {
    const promises = [this.sendPushNotification(payload), this.sendInAppNotification(payload)]

    await Promise.all(promises)
  }

  
  formatAlertMessage(segmentName: string, speed: number, volume: number, severity: string): string {
    const severityEmoji = {
      low: "⚠️",
      medium: "🟡",
      high: "🔴",
      critical: "🚨",
    }

    return `${severityEmoji[severity as keyof typeof severityEmoji]} ${segmentName}: Speed ${speed.toFixed(1)} km/h, Volume ${volume} vehicles. Severity: ${severity.toUpperCase()}`
  }
}
