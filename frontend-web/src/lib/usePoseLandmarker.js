import { useEffect, useState } from "react";
import { FilesetResolver, PoseLandmarker } from "@mediapipe/tasks-vision";

// The WASM version MUST match the installed @mediapipe/tasks-vision version (0.10.14 in package.json).
const WASM_URL = "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm";
const MODEL_URL =
  "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/1/pose_landmarker_full.task";

async function create(vision, delegate) {
  return PoseLandmarker.createFromOptions(vision, {
    baseOptions: { modelAssetPath: MODEL_URL, delegate },
    runningMode: "VIDEO",
    numPoses: 1,
  });
}

/** Loads Google's pre-trained pose model once (tries the GPU, falls back to CPU). */
export function usePoseLandmarker() {
  const [landmarker, setLandmarker] = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    let cancelled = false;
    let instance = null;
    (async () => {
      try {
        const vision = await FilesetResolver.forVisionTasks(WASM_URL);
        try {
          instance = await create(vision, "GPU");
        } catch {
          instance = await create(vision, "CPU");
        }
        if (cancelled) instance.close();
        else setLandmarker(instance);
      } catch (e) {
        if (!cancelled) setError(e.message || String(e));
      }
    })();
    return () => {
      cancelled = true;
      if (instance) instance.close();
    };
  }, []);

  return { landmarker, loading: !landmarker && !error, error };
}
