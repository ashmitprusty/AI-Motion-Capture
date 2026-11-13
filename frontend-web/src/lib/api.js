const API = (import.meta.env.VITE_API_URL ?? "http://localhost:8000").replace(/\/$/, "");

async function asJson(res) {
  if (!res.ok) {
    let detail = res.statusText;
    try {
      const body = await res.json();
      detail = typeof body.detail === "string" ? body.detail : JSON.stringify(body.detail);
    } catch {
      /* response had no JSON body */
    }
    throw new Error(detail || `Request failed (${res.status})`);
  }
  return res.json();
}

/** Phase 5, step 1: send the recorded landmark data to the Python backend. */
export async function uploadMotion(userName, recording) {
  const res = await fetch(`${API}/upload-motion-data`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ user_name: userName, ...recording }),
  });
  return asJson(res);
}

export async function getMotion(id) {
  return asJson(await fetch(`${API}/motions/${id}`));
}

export const absoluteUrl = (path) => `${API}${path}`;

/** Phase 5, step 2: poll until the backend has generated the animation file. */
export async function waitForMotion(id, { intervalMs = 1000, timeoutMs = 180000 } = {}) {
  const started = Date.now();
  for (;;) {
    const motion = await getMotion(id);
    if (motion.status === "done") return motion;
    if (motion.status === "error") throw new Error(motion.error || "Animation generation failed");
    if (Date.now() - started > timeoutMs) throw new Error("Timed out waiting for the animation");
    await new Promise((resolve) => setTimeout(resolve, intervalMs));
  }
}
