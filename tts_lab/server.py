from contextlib import asynccontextmanager
from pathlib import Path
import io
import json
import re
import uuid
from fastapi import FastAPI, File, UploadFile, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from dataclasses import asdict
from .config import ROOT, LoadConfig, saved_config, PROFILES
from .manager import Manager, BusyError

manager = Manager()
for folder in ('uploads', 'outputs', 'logs'):
    (ROOT / folder).mkdir(exist_ok=True)

@asynccontextmanager
async def lifespan(app):
    yield
    manager.stop()

app = FastAPI(title='TEST TTS — local laboratory', lifespan=lifespan)

@app.middleware('http')
async def local_only(request: Request, call_next):
    # Loopback server; reject browser requests from unrelated origins/DNS rebinding.
    host = request.headers.get('host', '').split(':')[0]
    if host not in ('127.0.0.1', 'localhost', 'testserver'):
        return JSONResponse({'detail': 'Local access only'}, status_code=403)
    origin = request.headers.get('origin')
    if origin and origin not in ('http://127.0.0.1:7860', 'http://localhost:7860'):
        return JSONResponse({'detail': 'Origin rejected'}, status_code=403)
    return await call_next(request)

@app.exception_handler(BusyError)
async def busy(request, exc):
    return JSONResponse({'detail': str(exc)}, status_code=409)

@app.exception_handler(ValueError)
async def bad_input(request, exc):
    return JSONResponse({'detail': str(exc)}, status_code=400)

@app.get('/')
def home():
    return FileResponse(ROOT / 'web' / 'index.html')

@app.get('/api/config')
def config():
    return {'config': asdict(saved_config()), 'profiles': PROFILES}

@app.get('/api/status')
def status():
    import psutil
    result = manager.snapshot()
    memory = psutil.virtual_memory()
    result['ram'] = {'used_gib': memory.used / 1024**3, 'available_gib': memory.available / 1024**3}
    return result

@app.get('/api/log')
def log():
    path = ROOT / 'logs' / 'worker.log'
    if not path.exists():
        return {'text': ''}
    with path.open('rb') as f:
        f.seek(max(0, path.stat().st_size - 12000))
        return {'text': f.read().decode('utf-8', errors='replace')}

class LoadRequest(BaseModel):
    profile: str = 'MOSS-TTS v1.5 8B'
    model_path: str
    codec_path: str
    placement: str = 'nf4'
    codec_device: str = 'cpu'
    gpu_gib: int = Field(11, ge=4, le=14)
    cpu_gib: int = Field(12, ge=4, le=20)
    threads: int = Field(8, ge=1, le=24)

@app.post('/api/load')
def load(req: LoadRequest):
    return manager.submit('load', config=LoadConfig(**req.model_dump()))

@app.post('/api/unload')
def unload():
    manager.stop()
    return {'ok': True}

class GenerateRequest(BaseModel):
    text: str = Field(min_length=1, max_length=10000)
    reference_id: str | None = None
    language: str = 'Persian (Farsi)'
    chunked: bool = True
    chunk_chars: int = Field(180, ge=40, le=500)
    max_new_tokens: int = Field(512, ge=64, le=4096)
    temperature: float = Field(1.7, ge=0, le=3)
    top_p: float = Field(0.8, gt=0, le=1)
    seed: int = Field(42, ge=0, le=2147483647)

@app.post('/api/generate')
def generate(req: GenerateRequest):
    if not req.text.strip():
        raise ValueError('متن خالی است.')
    reference = None
    if req.reference_id:
        if not re.fullmatch(r'[0-9a-f]{32}', req.reference_id):
            raise ValueError('Invalid reference ID')
        reference = ROOT / 'uploads' / (req.reference_id + '.wav')
        if not reference.is_file():
            raise ValueError('Reference file not found')
    options = req.model_dump(exclude={'text', 'reference_id'})
    return manager.submit('generate', text=req.text, reference=str(reference) if reference else None, options=options)

@app.post('/api/reference')
def reference(file: UploadFile = File(...)):
    import numpy as np
    import soundfile as sf
    data = file.file.read(20 * 1024 * 1024 + 1)
    if len(data) > 20 * 1024 * 1024:
        raise ValueError('Reference must be smaller than 20 MB')
    try:
        with sf.SoundFile(io.BytesIO(data)) as src:
            duration = len(src) / src.samplerate
            if not 1 <= duration <= 30:
                raise ValueError('Reference duration must be 1–30 seconds (recommended: 5–15 seconds).')
            if src.channels > 2:
                raise ValueError('Use mono or stereo audio')
            rate = src.samplerate
            wav = src.read(dtype='float32', always_2d=True).mean(axis=1)
    except (RuntimeError, TypeError) as exc:
        raise ValueError('Cannot read audio. Export a PCM WAV file.') from exc
    if not np.isfinite(wav).all() or np.max(np.abs(wav)) < 0.0001:
        raise ValueError('Reference is silent or invalid')
    rid = uuid.uuid4().hex
    sf.write(ROOT / 'uploads' / (rid + '.wav'), wav, rate, subtype='PCM_16')
    return {'id': rid, 'duration_s': duration}

@app.get('/audio/{job_id}/{filename}')
def audio(job_id: str, filename: str):
    if not re.fullmatch(r'[0-9a-f]{32}', job_id) or not re.fullmatch(r'(full|chunk-\d{3,})\.wav|metrics\.json', filename):
        raise HTTPException(404)
    path = ROOT / 'outputs' / job_id / filename
    if not path.is_file():
        raise HTTPException(404)
    return FileResponse(path)
