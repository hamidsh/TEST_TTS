import os
import time
import traceback
from dataclasses import asdict
from pathlib import Path


def run(commands, events, root):
    # Prevent nested HF loaders from fetching weights or Python remotely.
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['TRANSFORMERS_OFFLINE'] = '1'
    os.environ['HF_HUB_DISABLE_TELEMETRY'] = '1'
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    root = Path(root)
    (root / 'logs').mkdir(exist_ok=True)
    # Capture HF shard progress and tracebacks while the UI stays responsive.
    import sys
    log = open(root / 'logs' / 'worker.log', 'a', encoding='utf-8', buffering=1)
    sys.stdout = sys.stderr = log
    backend = None
    while True:
        job = commands.get()
        jid = job['id']
        def emit(message):
            print(message, flush=True)
            events.put({'id': jid, 'kind': 'progress', 'message': message})
        try:
            if job['action'] == 'load':
                from .config import LoadConfig
                from .moss import MossBackend
                cfg = LoadConfig(**job['config'])
                cfg.validate()
                start = time.perf_counter()
                backend = MossBackend()
                details = backend.load(cfg, emit)
                details['load_s'] = time.perf_counter() - start
                events.put({'id': jid, 'kind': 'done', 'result': details})
            else:
                if backend is None:
                    raise RuntimeError('Load a model first')
                import json
                import numpy as np
                import soundfile as sf
                import torch
                from .config import segments
                opts = job['options']
                torch.manual_seed(opts['seed'])
                if torch.cuda.is_available():
                    torch.cuda.reset_peak_memory_stats()
                parts = segments(job['text'], opts['chunk_chars']) if opts['chunked'] else [job['text']]
                total_start = time.perf_counter()
                folder = root / 'outputs' / jid
                folder.mkdir(parents=True, exist_ok=True)
                arrays, timings = [], []
                first = None
                for i, text in enumerate(parts):
                    emit(f'Chunk {i + 1}/{len(parts)}')
                    audio, rate, timing = backend.synthesize(text, job.get('reference'), opts, emit)
                    if first is None:
                        first = time.perf_counter() - total_start
                    name = f'chunk-{i + 1:03}.wav'
                    sf.write(folder / name, audio, rate, subtype='PCM_16')
                    arrays.append(audio)
                    timings.append(timing)
                    events.put({'id': jid, 'kind': 'chunk', 'url': f'/audio/{jid}/{name}', 'index': i})
                combined = np.concatenate(arrays)
                sf.write(folder / 'full.wav', combined, rate, subtype='PCM_16')
                elapsed = time.perf_counter() - total_start
                seconds = len(combined) / rate
                result = {'audio': f'/audio/{jid}/full.wav', 'elapsed_s': elapsed,
                          'first_chunk_s': first, 'audio_s': seconds, 'rtf': elapsed / seconds,
                          'chunks': timings, 'seed': opts['seed'],
                          'peak_vram_gib': torch.cuda.max_memory_allocated() / 1024**3 if torch.cuda.is_available() else 0,
                          'config': job['config'], 'options': opts,
                          'text': job['text'], 'reference_used': bool(job.get('reference'))}
                (folder / 'metrics.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
                events.put({'id': jid, 'kind': 'done', 'result': result})
        except Exception as exc:
            traceback.print_exc()
            events.put({'id': jid, 'kind': 'error', 'message': f'{type(exc).__name__}: {exc}'})
            # Exit on load failure: release partial allocations rather than retain a traceback/model.
            if job['action'] == 'load':
                return
