import { NextResponse } from "next/server";

export async function POST(req: Request) {
  try {
    const body = await req.json();
    
    const {
      target_column,
      task_type = "classification",
      test_size = 0.2,
      scaling_method = "standard",
      selection_priority = "balanced",
      project_name = "automl_project"
    } = body;

    if (!target_column) {
      return NextResponse.json(
        { error: "Target column is required" },
        { status: 400 }
      );
    }

    const backendRes = await fetch("http://127.0.0.1:8000/automl/train", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        target_column,
        task_type,
        test_size,
        scaling_method,
        selection_priority,
        project_name
      }),
    });

    const result = await backendRes.json();
    return NextResponse.json(result);
  } catch (err) {
    console.error("AutoML API Error:", err);
    return NextResponse.json(
      { error: "AutoML training failed" },
      { status: 500 }
    );
  }
}
