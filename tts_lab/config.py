from dataclasses import dataclass, asdict
from pathlib import Path
import json
import re

ROOT = Path(__file__).resolve().parent.parent
PROFILES = {
    'MOSS-TTS v1.5 8B': 'MOSS-TTS-v1.5',
    'MOSS-TTS 1.0 8B': 'MOSS-TTS',
}

@dataclass(frozen=True)
class LoadConfig:
    profile: str = 'MOSS-TTS v1.5 8B'
    model_path: str = str(ROOT / 'models' / 'MOSS-TTS-v1.5')
    codec_path: str = str(ROOT / 'models' / 'MOSS-Audio-Tokenizer')
    placement: str = 'nf4'
    codec_device: str = 'cpu'
    gpu_gib: int = 11
    cpu_gib: int = 12
    threads: int = 8

    def validate(self):
        if self.profile not in PROFILES:
            raise ValueError('Unknown model profile')
        if self.placement not in ('nf4', 'balanced', 'cpu'):
            raise ValueError('Unknown placement')
        if self.codec_device not in ('cpu', 'cuda'):
            raise ValueError('Unknown codec device')
        if not 4 <= self.gpu_gib <= 14 or not 4 <= self.cpu_gib <= 20 or not 1 <= self.threads <= 24:
            raise ValueError('Memory/thread setting outside supported range')
        validate_snapshot(self.model_path, True)
        validate_snapshot(self.codec_path, False)
        cfg = json.loads((Path(self.model_path) / 'config.json').read_text(encoding='utf-8'))
        if cfg.get('model_type') != 'moss_tts_delay':
            raise ValueError('This adapter requires MossTTSDelay (v1.5 8B or 1.0 8B), not a LoRA/GGUF/Realtime model.')


def validate_snapshot(folder, text_model):
    p = Path(folder)
    needed = ['config.json'] + (['tokenizer_config.json', 'processor_config.json', 'processing_moss_tts.py'] if text_model else [])
    missing = [f for f in needed if not (p / f).is_file()]
    indexes = list(p.glob('*.safetensors.index.json'))
    if indexes:
        for index in indexes:
            data = json.loads(index.read_text(encoding='utf-8'))
            for shard in set(data['weight_map'].values()):
                if not (p / shard).is_file():
                    missing.append(shard)
    elif not list(p.glob('*.safetensors')):
        missing.append('*.safetensors (all weight shards)')
    if text_model and not (p / 'tokenizer.json').exists():
        missing.append('tokenizer.json')
    if missing:
        raise ValueError(f'Incomplete local model: {p}\nMissing: ' + ', '.join(missing))


def segments(text, limit=180):
    """Bounded sentence chunks, preserving punctuation and Persian ZWNJ."""
    text = text.strip()
    if not text:
        raise ValueError('متن خالی است.')
    # Do not split inside explicit pause markers.
    parts = re.split(r'(?<=[.!؟?؛\n])\s+|\n+', text)
    result = []
    for part in parts:
        words = part.split()
        chunk = ''
        for word in words:
            if chunk and len(chunk) + len(word) + 1 > limit:
                result.append(chunk)
                chunk = ''
            chunk = (chunk + ' ' + word).strip()
        if chunk:
            result.append(chunk)
    return result


def saved_config():
    try:
        return LoadConfig(**json.loads((ROOT / 'settings.json').read_text(encoding='utf-8')))
    except (OSError, ValueError, TypeError):
        return LoadConfig()


def save_config(config):
    temp = ROOT / 'settings.json.tmp'
    temp.write_text(json.dumps(asdict(config), ensure_ascii=False, indent=2), encoding='utf-8')
    temp.replace(ROOT / 'settings.json')
