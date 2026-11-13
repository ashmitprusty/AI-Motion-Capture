"""AI Motion Capture - FastAPI backend.

Run:  uvicorn server:app --reload --port 8000
Docs: http://localhost:8000/docs
"""
import logging
import os
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

from bson import ObjectId
from bson.errors import InvalidId
from dotenv import load_dotenv
from fastapi import BackgroundTasks, FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field, model_validator
from pymongo import DESCENDING, MongoClient

from mocap.landmarks import NUM_LANDMARKS
from mocap.pipeline import MIN_FRAMES, process_motion

load_dotenv()
log = logging.getLogger("mocap")
logging.basicConfig(level=logging.INFO)

MONGODB_URI = os.getenv("MONGODB_URI")
if not MONGODB_URI:
    raise RuntimeError("MONGODB_URI is not set. Copy .env.example to .env and fill it in.")
MONGODB_DB = os.getenv("MONGODB_DB", "motion_capture")
OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", "outputs")).resolve()
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
TARGET_FPS = int(os.getenv("TARGET_FPS", "30"))
CORS_ORIGINS = [o.strip() for o in os.getenv("CORS_ORIGINS", "http://localhost:5173").split(",") if o.strip()]
MAX_FRAMES = 4000          # about 60 s of camera footage at up to ~60 fps

client = MongoClient(MONGODB_URI, serverSelectionTimeoutMS=8000)
motions = client[MONGODB_DB]["motions"]


@asynccontextmanager
async def lifespan(_: FastAPI):
    try:
        motions.create_index([("user_name", 1), ("created_at", DESCENDING)])
        log.info("Connected to MongoDB")
    except Exception as exc:  # the API still starts so /health can report the problem
        log.warning("MongoDB not reachable at start-up: %s", exc)
    yield
    client.close()


app = FastAPI(title="AI Motion Capture API", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=CORS_ORIGINS, allow_methods=["*"], allow_headers=["*"])


class MotionUpload(BaseModel):
    user_name: str = Field(min_length=1, max_length=80)
    video_width: int = Field(gt=0)
    video_height: int = Field(gt=0)
    timestamps: List[float]                      # seconds since recording started
    world: List[List[List[float]]]               # [frame][33][x, y, z, visibility]  (metres, hip-centred)
    image: List[List[List[float]]]               # [frame][33][x, y, z, visibility]  (normalised 0..1)

    @model_validator(mode="after")
    def check_shapes(self):
        n = len(self.timestamps)
        if n < MIN_FRAMES:
            raise ValueError(f"at least {MIN_FRAMES} frames are required")
        if n > MAX_FRAMES:
            raise ValueError(f"at most {MAX_FRAMES} frames are allowed")
        if len(self.world) != n or len(self.image) != n:
            raise ValueError("timestamps, world and image must have the same number of frames")
        for frames in (self.world, self.image):
            for frame in frames:
                if len(frame) != NUM_LANDMARKS or any(len(p) != 4 for p in frame):
                    raise ValueError(f"every frame needs {NUM_LANDMARKS} landmarks of [x, y, z, visibility]")
        return self


def _object_id(motion_id: str) -> ObjectId:
    try:
        return ObjectId(motion_id)
    except (InvalidId, TypeError):
        raise HTTPException(status_code=404, detail="Motion not found")


def _public(doc: dict) -> dict:
    motion_id = str(doc["_id"])
    done = doc.get("status") == "done"
    return {
        "id": motion_id,
        "user_name": doc.get("user_name"),
        "created_at": doc["created_at"].isoformat() if doc.get("created_at") else None,
        "status": doc.get("status"),
        "error": doc.get("error"),
        "stats": doc.get("stats"),
        "download_url": f"/motions/{motion_id}/bvh" if done else None,
    }


def run_generation(motion_id: str) -> None:
    """Background job: raw landmarks in MongoDB -> cleaned -> rotations -> .bvh file."""
    oid = ObjectId(motion_id)
    try:
        doc = motions.find_one({"_id": oid})
        stats = process_motion(doc, OUTPUT_DIR / f"{motion_id}.bvh", TARGET_FPS)
        motions.update_one({"_id": oid}, {"$set": {"status": "done", "stats": stats, "error": None}})
        log.info("Motion %s generated: %s", motion_id, stats)
    except Exception as exc:
        log.exception("Generation failed for %s", motion_id)
        motions.update_one({"_id": oid}, {"$set": {"status": "error", "error": str(exc)}})


@app.get("/health")
def health():
    try:
        client.admin.command("ping")
        return {"ok": True, "database": "connected"}
    except Exception as exc:
        return {"ok": False, "database": f"unreachable: {exc}"}


@app.post("/upload-motion-data", status_code=202)
def upload_motion_data(payload: MotionUpload, background: BackgroundTasks):
    doc = payload.model_dump()
    doc.update({"created_at": datetime.now(timezone.utc), "status": "processing", "error": None, "stats": None})
    motion_id = str(motions.insert_one(doc).inserted_id)
    background.add_task(run_generation, motion_id)
    return {"id": motion_id, "status": "processing"}


@app.get("/motions")
def list_motions(user_name: Optional[str] = None, limit: int = Query(20, ge=1, le=100)):
    query = {"user_name": user_name} if user_name else {}
    cursor = motions.find(query, {"world": 0, "image": 0}).sort("created_at", DESCENDING).limit(limit)
    return [_public(d) for d in cursor]


@app.get("/motions/{motion_id}")
def get_motion(motion_id: str):
    doc = motions.find_one({"_id": _object_id(motion_id)}, {"world": 0, "image": 0})
    if doc is None:
        raise HTTPException(status_code=404, detail="Motion not found")
    return _public(doc)


@app.get("/motions/{motion_id}/bvh")
def download_bvh(motion_id: str):
    oid = _object_id(motion_id)
    doc = motions.find_one({"_id": oid}, {"status": 1})
    if doc is None:
        raise HTTPException(status_code=404, detail="Motion not found")
    path = OUTPUT_DIR / f"{oid}.bvh"
    if doc.get("status") != "done" or not path.exists():
        raise HTTPException(status_code=409, detail="Animation is not ready yet")
    return FileResponse(path, media_type="application/octet-stream", filename=f"motion_{oid}.bvh")
