"""MossTTSDelay adapter. Heavy imports happen only in the isolated worker."""
from pathlib import Path
import hashlib
import time


class MossBackend:
    def __init__(self):
        self.model = self.processor = None
        self.reference_cache = {}

    def load(self, cfg, emit):
        import torch
        from transformers import AutoModel, AutoProcessor, BitsAndBytesConfig
        torch.set_num_threads(cfg.threads)
        if (cfg.placement != 'cpu' or cfg.codec_device == 'cuda') and not torch.cuda.is_available():
            raise RuntimeError('CUDA is unavailable. Install the CUDA torch build or select CPU for both components.')
        common = dict(trust_remote_code=True, local_files_only=True)
        emit('Loading audio codec + text tokenizer …')
        # These kwargs apply ONLY to the codec, never to the language model.
        self.processor = AutoProcessor.from_pretrained(
            cfg.model_path, codec_path=cfg.codec_path,
            torch_dtype=torch.float32 if cfg.codec_device == 'cpu' else torch.bfloat16,
            device_map={'': cfg.codec_device}, **common,
        )
        self.processor.audio_tokenizer.eval()
        emit('Loading language model shards … (first load can take several minutes)')
        kwargs = dict(torch_dtype=torch.bfloat16, attn_implementation='sdpa', low_cpu_mem_usage=True)
        if cfg.placement == 'nf4':
            kwargs.update(device_map={'': 0}, quantization_config=BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type='nf4',
                bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16,
            ))
        elif cfg.placement == 'balanced':
            # Leave VRAM for KV cache, activations and desktop. Codec is separate.
            kwargs.update(device_map='auto', max_memory={0: f'{cfg.gpu_gib}GiB', 'cpu': f'{cfg.cpu_gib}GiB'})
        else:
            kwargs.update(device_map={'': 'cpu'}, torch_dtype=torch.bfloat16)
        self.model = AutoModel.from_pretrained(cfg.model_path, **common, **kwargs).eval()
        mapping = getattr(self.model, 'hf_device_map', {})
        if 'disk' in mapping.values():
            raise RuntimeError('Disk offload is disabled. Use NF4 or adjust memory budgets.')
        self.sample_rate = int(getattr(self.processor.model_config, 'sampling_rate', 24000))
        self.device = next(self.model.parameters()).device
        if self.device.type == 'meta':
            self.device = torch.device('cuda:0')
        return {'device_map': {k: str(v) for k, v in mapping.items()},
                'codec_device': str(next(self.processor.audio_tokenizer.parameters()).device),
                'text_tokenizer': 'CPU', 'sample_rate': self.sample_rate}

    def synthesize(self, text, reference, options, emit):
        import torch
        started = time.perf_counter()
        reference_codes = None
        with torch.inference_mode():
            if reference:
                data = Path(reference).read_bytes()
                key = hashlib.sha256(data).hexdigest()
                if key not in self.reference_cache:
                    emit('Encoding reference voice …')
                    # WAV is normalized by the upload endpoint. Cache codes, not a transcript.
                    import soundfile as sf
                    wav, rate = sf.read(reference, dtype='float32', always_2d=True)
                    waveform = torch.from_numpy(wav.mean(axis=1)).unsqueeze(0)
                    codes = self.processor.encode_audios_from_wav([waveform], rate)[0]
                    self.reference_cache.clear()  # bounded: keep only the most recent voice
                    self.reference_cache[key] = codes.detach().cpu()
                reference_codes = [self.reference_cache[key]]
            message = self.processor.build_user_message(
                text=text, reference=reference_codes, language=options['language'] or None)
            batch = self.processor([[message]], mode='generation')
            prepare = time.perf_counter() - started
            emit('Generating audio tokens …')
            generation_start = time.perf_counter()
            output = self.model.generate(
                input_ids=batch['input_ids'].to(self.device),
                attention_mask=batch['attention_mask'].to(self.device),
                max_new_tokens=options['max_new_tokens'],
                audio_temperature=options['temperature'], audio_top_p=options['top_p'],
                audio_top_k=25, audio_repetition_penalty=1.0,
            )
            generation = time.perf_counter() - generation_start
            # Custom MOSS generate returns [(prompt_length, token_tensor)], not a tensor.
            prompt_length, token_sequence = output[0]
            truncated = token_sequence.shape[0] - prompt_length >= options['max_new_tokens']
            emit('Decoding waveform …')
            decode_start = time.perf_counter()
            messages = self.processor.decode(output)
            if not messages or not messages[0] or not messages[0].audio_codes_list:
                raise RuntimeError('Model returned no decodable audio. Try a shorter sentence or a higher token limit.')
            audio = messages[0].audio_codes_list[0].detach().float().cpu().numpy().reshape(-1)
            if not len(audio):
                raise RuntimeError('Model returned empty audio')
            return audio, self.sample_rate, {
                'prepare_s': prepare, 'generate_s': generation,
                'decode_s': time.perf_counter() - decode_start,
                'token_limit_reached': truncated,
            }
