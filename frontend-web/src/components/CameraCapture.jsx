import { useCallback, useEffect, useRef, useState } from "react";
import Webcam from "react-webcam";
import { DrawingUtils, PoseLandmarker } from "@mediapipe/tasks-vision";
import { usePoseLandmarker } from "../lib/usePoseLandmarker";

const MAX_SECONDS = 60;
const MIN_FRAMES = 15;
const COUNTDOWN_SECONDS = 3;

const round4 = (n) => Math.round(n * 10000) / 10000;
const pack = (landmarks) => landmarks.map((p) => [round4(p.x), round4(p.y), round4(p.z), round4(p.visibility ?? 0)]);

/**
 * Phase 2: live webcam + skeleton overlay + recording of every detected frame.
 * Calls onRecordingComplete({ timestamps, world, image, video_width, video_height }).
 */
export default function CameraCapture({ disabled, onRecordingComplete }) {
  const webcamRef = useRef(null);
  const canvasRef = useRef(null);
  const { landmarker, loading, error } = usePoseLandmarker();

  const [cameraReady, setCameraReady] = useState(false);
  const [cameraError, setCameraError] = useState("");
  const [phase, setPhase] = useState("idle"); // idle | countdown | recording
  const [countdown, setCountdown] = useState(COUNTDOWN_SECONDS);
  const [elapsed, setElapsed] = useState(0);
  const [frameCount, setFrameCount] = useState(0);
  const [message, setMessage] = useState("");

  // Refs hold values that the animation loop must read without re-rendering.
  const phaseRef = useRef("idle");
  const recRef = useRef(null); // { startMs, timestamps, world, image }
  const countdownTimer = useRef(null);
  const completeRef = useRef(onRecordingComplete);
  useEffect(() => {
    completeRef.current = onRecordingComplete;
  }, [onRecordingComplete]);

  const setPhaseBoth = (next) => {
    phaseRef.current = next;
    setPhase(next);
  };

  // ---- detection + drawing loop (runs every video frame) ----
  useEffect(() => {
    if (!landmarker || !cameraReady) return undefined;
    const canvas = canvasRef.current;
    const ctx = canvas.getContext("2d");
    const drawing = new DrawingUtils(ctx);
    let raf = 0;
    let lastVideoTime = -1;

    const loop = () => {
      const video = webcamRef.current?.video;
      if (video && video.readyState >= 2 && video.currentTime !== lastVideoTime) {
        lastVideoTime = video.currentTime;
        if (canvas.width !== video.videoWidth || canvas.height !== video.videoHeight) {
          canvas.width = video.videoWidth;
          canvas.height = video.videoHeight;
        }
        const now = performance.now();
        const result = landmarker.detectForVideo(video, now);
        ctx.clearRect(0, 0, canvas.width, canvas.height);

        const image = result.landmarks[0];
        const world = result.worldLandmarks[0];
        if (image && world) {
          // Phase 2, step "Draw the Skeleton"
          drawing.drawConnectors(image, PoseLandmarker.POSE_CONNECTIONS, { color: "#22d3ee", lineWidth: 3 });
          drawing.drawLandmarks(image, { color: "#f43f5e", radius: 3 });

          // Phase 2, step "Capture the Data": every frame, not once per second
          const rec = recRef.current;
          if (phaseRef.current === "recording" && rec) {
            rec.timestamps.push(round4((now - rec.startMs) / 1000));
            rec.world.push(pack(world)); // metres, centred on the hips
            rec.image.push(pack(image)); // 0..1 inside the picture
          }
        }
      }
      raf = requestAnimationFrame(loop);
    };
    raf = requestAnimationFrame(loop);
    return () => cancelAnimationFrame(raf);
  }, [landmarker, cameraReady]);

  // ---- stop / finish ----
  const stopRecording = useCallback(() => {
    const rec = recRef.current;
    recRef.current = null;
    setPhaseBoth("idle");
    if (!rec) return;
    if (rec.timestamps.length < MIN_FRAMES) {
      setMessage("Not enough frames with a detected body. Stand fully in view with good lighting and try again.");
      return;
    }
    const video = webcamRef.current.video;
    completeRef.current({
      timestamps: rec.timestamps,
      world: rec.world,
      image: rec.image,
      video_width: video.videoWidth,
      video_height: video.videoHeight,
    });
  }, []);

  // live counters + automatic stop at the time limit
  useEffect(() => {
    if (phase !== "recording") return undefined;
    const id = setInterval(() => {
      const rec = recRef.current;
      if (!rec) return;
      const seconds = (performance.now() - rec.startMs) / 1000;
      setElapsed(seconds);
      setFrameCount(rec.timestamps.length);
      if (seconds >= MAX_SECONDS) stopRecording();
    }, 250);
    return () => clearInterval(id);
  }, [phase, stopRecording]);

  useEffect(() => () => clearInterval(countdownTimer.current), []);

  // ---- start ----
  const startRecording = () => {
    setMessage("");
    setElapsed(0);
    setFrameCount(0);
    let n = COUNTDOWN_SECONDS;
    setCountdown(n);
    setPhaseBoth("countdown");
    countdownTimer.current = setInterval(() => {
      n -= 1;
      if (n <= 0) {
        clearInterval(countdownTimer.current);
        recRef.current = { startMs: performance.now(), timestamps: [], world: [], image: [] };
        setPhaseBoth("recording");
      } else {
        setCountdown(n);
      }
    }, 1000);
  };

  const cancelCountdown = () => {
    clearInterval(countdownTimer.current);
    setPhaseBoth("idle");
  };

  const modelStatus = error ? `Model failed to load: ${error}` : loading ? "Loading AI model…" : "";

  return (
    <div>
      <div className="video-wrap">
        {/* The wrapper is flipped with CSS so it behaves like a mirror; the recorded data is NOT flipped. */}
        <div className="mirror">
          <Webcam
            ref={webcamRef}
            audio={false}
            mirrored={false}
            videoConstraints={{ width: 640, height: 480, facingMode: "user" }}
            onUserMedia={() => setCameraReady(true)}
            onUserMediaError={(e) => setCameraError(e?.message || "Camera permission denied")}
          />
          <canvas ref={canvasRef} className="overlay" />
        </div>
        {phase === "countdown" && <div className="badge big">{countdown}</div>}
        {phase === "recording" && <div className="badge rec">● REC {elapsed.toFixed(1)}s · {frameCount} frames</div>}
      </div>

      {(modelStatus || cameraError) && <p className="note warn">{cameraError || modelStatus}</p>}
      {message && <p className="note warn">{message}</p>}

      <div className="row">
        {phase === "idle" && (
          <button className="btn primary" disabled={disabled || !landmarker || !cameraReady} onClick={startRecording}>
            Start recording
          </button>
        )}
        {phase === "countdown" && (
          <button className="btn" onClick={cancelCountdown}>
            Cancel
          </button>
        )}
        {phase === "recording" && (
          <button className="btn danger" onClick={stopRecording}>
            Stop &amp; generate animation
          </button>
        )}
      </div>
      <p className="note">Step back so your whole body is visible. Max {MAX_SECONDS} seconds per recording.</p>
    </div>
  );
}
