import { NextRequest, NextResponse } from "next/server";

const BACKEND_URL = process.env.NEXT_PUBLIC_BACKEND_URL || "http://localhost:8000";

// Increase timeout for explainability generation
export const maxDuration = 300;

export async function POST(req: NextRequest) {
  try {
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), 300000); // 5 minutes

    // Get the request body
    const body = await req.json();

    const response = await fetch(`${BACKEND_URL}/automl/explain-tabular`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify(body),
      signal: controller.signal,
    });

    clearTimeout(timeoutId);

    const data = await response.json();
    return NextResponse.json(data);
  } catch (error) {
    console.error("Explainability API error:", error);
    return NextResponse.json(
      { status: "error", error: String(error) },
      { status: 500 }
    );
  }
}
