# server.py
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
from PIL import Image
import requests
import asyncio
import torch
from io import BytesIO

import model_loader
from free_space_polygon import loc_to_json, process_kins_site

########## FOR TESTING
import visualizer

# ── Startup / Shutdown ───────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Runs BEFORE first request — model loads here
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, model_loader.load)
    yield
    # Runs on shutdown — cleanup if needed
    print("[Server] Shutting down")

app = FastAPI(lifespan=lifespan)

NODE_WEBHOOK_URL = "http://your-node-server.com/api/receive-kins-data"


# ── Request schema ───────────────────────────────────────────────────────────
class KinsInput(BaseModel):
    success:     bool
    lat:         float
    lng:         float
    coordinates: dict   # {x: int, y: int}
    imagePath:   str    # URL or local path


# ── Image loader ─────────────────────────────────────────────────────────────
def _load_image(path: str) -> Image.Image:
    if path.startswith('http://') or path.startswith('https://'):
        resp = requests.get(path, timeout=15)
        resp.raise_for_status()
        return Image.open(BytesIO(resp.content)).convert('RGB')
    return Image.open(path).convert('RGB')


# ── Full pipeline (runs in thread pool) ──────────────────────────────────────
def _run_pipeline(payload: KinsInput) -> dict:
    image       = _load_image(payload.imagePath)
    detections  = model_loader.infer(image)        # read-only inference
    visualizer.draw_raw_detections(payload.imagePath, detections)
    
    # 2. Convert and Process
    object_list = loc_to_json(detections)
    result = process_kins_site(object_list, payload.model_dump())
    
    # Visual Debug: Save final polygons
    if result:
        visualizer.draw_final_polygons(result)

    if result is None:
        return {
            'success': False,
            'error':   'No rooftop detected at clicked coordinates',
            'lat':     payload.lat,
            'lng':     payload.lng,
        }

    return {'success': True, 'lat': payload.lat, 'lng': payload.lng, **result}


# ── Background task — runs pipeline then pushes to Node ──────────────────────
async def _process_and_respond(payload: KinsInput):
    loop   = asyncio.get_event_loop()
    result = await loop.run_in_executor(None, _run_pipeline, payload)

    try:
        requests.post(NODE_WEBHOOK_URL, json=result, timeout=10)
        print(f"[Server] Pushed result to Node — success={result['success']}")
    except Exception as e:
        print(f"[Server] Failed to push to Node: {e}")


# ── Endpoints ─────────────────────────────────────────────────────────────────
@app.post('/analyze-roof')
async def analyze_roof(payload: KinsInput, background_tasks: BackgroundTasks):
    if not payload.success:
        raise HTTPException(status_code=400, detail="Payload marked as failed")

    if not model_loader.is_ready():
        raise HTTPException(status_code=503, detail="Model not ready yet")

    # background_tasks.add_task(_process_and_respond, payload)
    # return {'success': True, 'message': 'Analysis started'}

    try:
        # Run the pipeline synchronously to see the result in your terminal/client
        result = _run_pipeline(payload)
        
        if not result['success']:
            return {"status": "error", "message": result.get('error')}
            
        print(f"[LocalTest] Analysis Complete for {payload.lat}, {payload.lng}")
        return result # This will show up in your Postman/Browser
        
    except Exception as e:
        print(f"[LocalTest] Pipeline failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get('/health')
def health():
    return {
        'status':       'ready' if model_loader.is_ready() else 'loading',
        'device':       'cuda' if torch.cuda.is_available() else 'cpu',
        'vram_used_gb': round(torch.cuda.memory_allocated() / 1e9, 2)
                        if torch.cuda.is_available() else None,
    }

