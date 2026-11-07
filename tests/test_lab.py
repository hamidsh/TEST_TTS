import io
import json
import time
from unittest.mock import patch
import numpy as np
import pytest
import soundfile as sf
from fastapi.testclient import TestClient
from tts_lab.config import LoadConfig, segments, validate_snapshot
from tts_lab.manager import Manager, BusyError
from tts_lab.server import app


def snapshot(tmp_path):
    p = tmp_path / 'model'
    p.mkdir()
    for name in ('tokenizer.json', 'tokenizer_config.json', 'processor_config.json', 'config.json'):
        (p / name).write_text(json.dumps({'model_type': 'moss_tts_delay'}))
    (p / 'processing_moss_tts.py').write_text('# fixture')
    (p / 'a.safetensors').touch()
    return p


def test_shard_validation(tmp_path):
    p = snapshot(tmp_path)
    (p / 'model.safetensors.index.json').write_text(json.dumps({'weight_map': {'a': 'a.safetensors', 'b': 'missing.safetensors'}}))
    with pytest.raises(ValueError, match='missing.safetensors'):
        validate_snapshot(str(p), True)
    (p / 'missing.safetensors').touch()
    validate_snapshot(str(p), True)


def test_persian_chunks():
    text = 'سلام دنیا! این یک جمله است. می\u200cخواهم فارسی صحبت کنم؟ جمله آخر'
    parts = segments(text, 20)
    assert ' '.join(parts) == text
    assert all(len(p) <= 20 for p in parts)
    assert any('می\u200cخواهم' in p for p in parts)
    with pytest.raises(ValueError):
        segments('   ')


def fake_worker(commands, events, root):
    while True:
        job = commands.get()
        if job['action'] == 'load':
            events.put({'id': job['id'], 'kind': 'done', 'result': {'fake': True}})
        else:
            events.put({'id': job['id'], 'kind': 'progress', 'message': 'working'})
            time.sleep(30)


def test_worker_lifecycle(tmp_path):
    p = snapshot(tmp_path)
    cfg = LoadConfig(model_path=str(p), codec_path=str(p))
    m = Manager()
    with patch('tts_lab.manager.run', fake_worker), patch('tts_lab.manager.save_config'):
        try:
            m.submit('load', config=cfg)
            deadline = time.monotonic() + 10
            while m.snapshot()['active'] and time.monotonic() < deadline:
                time.sleep(.05)
            assert m.snapshot()['loaded']
            pid = m.snapshot()['worker_pid']
            assert m.submit('load', config=cfg) == {'already_loaded': True}
            assert m.snapshot()['worker_pid'] == pid
            m.submit('generate', text='سلام', options={})
            with pytest.raises(BusyError):
                m.submit('generate', text='سلام', options={})
            m.stop()
            s = m.snapshot()
            assert s['loaded'] is None and s['worker_pid'] is None
            assert s['jobs'][-1]['state'] == 'cancelled'
        finally:
            m.stop()


def test_api_validation_and_reference(tmp_path):
    import tts_lab.server as server
    (tmp_path / 'uploads').mkdir()
    with patch.object(server, 'ROOT', tmp_path), TestClient(app) as client:
        assert client.post('/api/generate', json={'text': ' '}).status_code == 400
        assert client.post('/api/generate', json={'text': 'سلام'}).status_code == 400
        assert client.post('/api/generate', json={'text': 'سلام', 'reference_id': '../../x'}).status_code == 400
        assert client.get('/api/status', headers={'origin': 'https://evil.example'}).status_code == 403
        assert client.get('/api/status', headers={'host': 'evil.example'}).status_code == 403
        assert client.post('/api/generate', json={'text': 'سلام', 'max_new_tokens': 0}).status_code == 422
        assert client.post('/api/reference', files={'file': ('broken.wav', b'bad')}).status_code == 400
        audio = io.BytesIO()
        sf.write(audio, np.sin(np.arange(48000) * .03) * .1, 24000, format='WAV')
        response = client.post('/api/reference', files={'file': ('voice.wav', audio.getvalue())})
        assert response.status_code == 200
        assert response.json()['duration_s'] == 2
        assert (tmp_path / 'uploads' / (response.json()['id'] + '.wav')).is_file()
        short = io.BytesIO()
        sf.write(short, np.zeros(100), 24000, format='WAV')
        assert client.post('/api/reference', files={'file': ('short.wav', short.getvalue())}).status_code == 400
