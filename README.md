
# AI Motion Capture

Webcam → MediaPipe body tracking (React) → FastAPI + MongoDB Atlas → cleaning, joint rotations (numpy/scipy) → `.bvh` animation → live 3D preview (three.js).

```
AI-Motion-Capture/
├── frontend-web/            React (Vite) app: camera, skeleton overlay, recorder, 3D viewer
│   └── src/
│       ├── App.jsx                       page layout + upload/poll flow
│       ├── components/CameraCapture.jsx  webcam, MediaPipe, skeleton drawing, recording
│       ├── components/MotionViewer.jsx   three.js player for the generated .bvh
│       └── lib/{api.js, usePoseLandmarker.js}
└── backend-api/             FastAPI server + motion pipeline
    ├── server.py                         API routes + MongoDB
    ├── mocap/
    │   ├── landmarks.py                  MediaPipe indices, axis conversion
    │   ├── smoothing.py                  gap fill, resample, despike, Kalman
    │   ├── skeleton.py                   the T-pose humanoid rig
    │   ├── rotations.py                  positions → joint rotations
    │   ├── bvh_export.py                 .bvh writer
    │   └── pipeline.py                   runs all of the above
    ├── tests/test_pipeline.py            offline test with a synthetic person
    └── tools/bvh_to_fbx.py               optional Blender converter
```

## Phase 1 — Setup
1. Install VS Code, Node.js (LTS), Python 3.10+, Git.
2. Create the GitHub repo; every teammate runs `git clone <repo-url>`.
3. Folders already exist: `frontend-web/` and `backend-api/`.
4. MongoDB Atlas: create a free cluster → Database Access (create user) → Network Access (add your IP) → Connect → Drivers → copy the connection string.

## Phase 3 — Backend (start it first)
```bash
cd backend-api
python -m venv .venv
# Windows: .venv\Scripts\activate      macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env        # Windows: copy .env.example .env   -> then paste your MONGODB_URI
python tests/test_pipeline.py          # should print PASS (no camera or database needed)
uvicorn server:app --reload --port 8000
```
Open http://localhost:8000/health (should say connected) and http://localhost:8000/docs.

## Phase 2 + 5 — Frontend
```bash
cd frontend-web
npm install
npm run dev
```
Open http://localhost:5173 (Chrome/Edge), allow the camera, enter a name, record, stop.

## Optional — FBX
`blender --background --python backend-api/tools/bvh_to_fbx.py -- backend-api/outputs/<id>.bvh out.fbx`

## Known limits (be upfront about these in your report)
- MediaPipe gives joint positions only: no forearm/wrist twist, no finger tracking, one spine segment.
- Feet are not locked to the floor (some foot sliding); depth travel (towards/away from camera) is not recovered.
- Single person, fixed camera, good light, full body in frame.
