import { useCallback, useState } from "react";
import CameraCapture from "./components/CameraCapture.jsx";
import MotionViewer from "./components/MotionViewer.jsx";
import { absoluteUrl, uploadMotion, waitForMotion } from "./lib/api.js";

const STATUS_TEXT = {
  idle: "Enter your name, then record a short movement.",
  uploading: "Uploading your motion data…",
  processing: "Cleaning data and generating the animation on the server…",
  done: "Animation ready!",
  error: "Something went wrong.",
};

export default function App() {
  const [userName, setUserName] = useState("");
  const [status, setStatus] = useState("idle");
  const [motion, setMotion] = useState(null);
  const [error, setError] = useState("");

  const busy = status === "uploading" || status === "processing";

  const handleRecording = useCallback(
    async (recording) => {
      setError("");
      setMotion(null);
      setStatus("uploading");
      try {
        const { id } = await uploadMotion(userName.trim(), recording);
        setStatus("processing");
        setMotion(await waitForMotion(id));
        setStatus("done");
      } catch (e) {
        setError(e.message);
        setStatus("error");
      }
    },
    [userName],
  );

  const bvhUrl = motion?.download_url ? absoluteUrl(motion.download_url) : null;

  return (
    <div className="page">
      <header>
        <h1>AI Motion Capture</h1>
        <p>Record your movement with a webcam and turn it into a 3D game animation.</p>
      </header>

      <main className="grid">
        <section className="card">
          <h2>1 · Record</h2>
          <label className="field">
            Your name
            <input
              value={userName}
              onChange={(e) => setUserName(e.target.value)}
              placeholder="e.g. Ashmit"
              maxLength={80}
              disabled={busy}
            />
          </label>
          <CameraCapture disabled={!userName.trim() || busy} onRecordingComplete={handleRecording} />
          <p className={`status ${status}`}>{STATUS_TEXT[status]}</p>
          {error && <p className="note warn">{error}</p>}
        </section>

        <section className="card">
          <h2>2 · 3D preview</h2>
          {bvhUrl ? (
            <>
              <MotionViewer url={bvhUrl} />
              <div className="row">
                <a className="btn primary" href={bvhUrl}>
                  Download .bvh
                </a>
                <span className="note">
                  {motion.stats.duration_s}s · {motion.stats.frames} frames @ {motion.stats.fps} fps
                </span>
              </div>
              <p className="note">Drag to orbit, scroll to zoom. Import the .bvh into Blender, Unity or Unreal.</p>
            </>
          ) : (
            <div className="viewer placeholder">Your animation will appear here after you record.</div>
          )}
        </section>
      </main>
    </div>
  );
}
