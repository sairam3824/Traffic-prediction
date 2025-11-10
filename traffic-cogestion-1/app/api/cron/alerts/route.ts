import { AlertGenerationService } from "@/lib/services/alert-generation-service"
import { NextResponse } from "next/server"


export async function GET(request: Request) {
  try {
    
    const authHeader = request.headers.get("authorization")
    if (authHeader !== `Bearer ${process.env.CRON_SECRET}`) {
      return NextResponse.json({ success: false, error: "Unauthorized" }, { status: 401 })
    }

    const alertService = new AlertGenerationService()

    
    await alertService.checkPredictionsAndGenerateAlerts()

    return NextResponse.json({
      success: true,
      message: "Cron job executed successfully",
      timestamp: new Date().toISOString(),
    })
  } catch (error) {
    console.error("Error in cron job:", error)
    return NextResponse.json(
      {
        success: false,
        error: "Cron job failed",
      },
      { status: 500 },
    )
  }
}
