import { NextResponse } from "next/server";

export async function POST(req: Request) {
  try {
    const body = await req.json();
    
    const {
      data,
      target_column = null,
      protected_columns = []
    } = body;

    if (!data || !Array.isArray(data) || data.length === 0) {
      return NextResponse.json(
        { error: "Data is required" },
        { status: 400 }
      );
    }

    const backendRes = await fetch("http://127.0.0.1:8000/data/clean", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        data,
        target_column,
        protected_columns
      }),
    });

    const result = await backendRes.json();
    return NextResponse.json(result);
  } catch (err) {
    console.error("Data cleaning API error:", err);
    return NextResponse.json(
      { error: "Data cleaning failed" },
      { status: 500 }
    );
  }
}
