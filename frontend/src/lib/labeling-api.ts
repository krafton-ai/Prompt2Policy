const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

export interface LabelingStatus {
  enabled: boolean;
  annotator: string;
}

export interface HumanLabelRequest {
  session_id: string;
  iteration: number;
  annotator: string;
  intent_score: number;
  video_url: string;
}

export interface HumanLabelResponse {
  status: string;
  video_count: number;
}

export async function fetchLabelingStatus(): Promise<LabelingStatus> {
  const res = await fetch(`${API_BASE}/api/human-label/status`);
  if (!res.ok) throw new Error(`API error: ${res.status}`);
  return res.json();
}

export async function submitHumanLabel(
  req: HumanLabelRequest,
): Promise<HumanLabelResponse> {
  const res = await fetch(`${API_BASE}/api/human-label`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(req),
  });
  if (!res.ok) throw new Error(`API error: ${res.status}`);
  return res.json();
}
