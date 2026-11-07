"""Regression for Unexpected keyword argument torch_dtype (no real weights needed)."""
import json
import sys
from types import ModuleType
from unittest.mock import Mock
import pytest
from tts_lab.moss import load_local_processor


@pytest.mark.parametrize('device,dtype', [('cpu', 'float32'), ('cuda', 'bfloat16')])
def test_load_options_never_reach_processor_constructor(tmp_path, monkeypatch, device, dtype):
    (tmp_path / 'processor_config.json').write_text(json.dumps({
        'auto_map': {'AutoProcessor': 'processing_moss_tts.MossTTSDelayProcessor'}
    }))
    transformers = ModuleType('transformers')
    dynamic = ModuleType('transformers.dynamic_module_utils')
    config, tokenizer, codec = object(), object(), Mock()
    codec.eval.return_value = codec
    transformers.AutoConfig = Mock()
    transformers.AutoTokenizer = Mock()
    transformers.AutoModel = Mock()
    transformers.AutoConfig.from_pretrained.return_value = config
    transformers.AutoTokenizer.from_pretrained.return_value = tokenizer
    transformers.AutoModel.from_pretrained.return_value = codec

    class StrictProcessor:
        # Any leaked torch_dtype/device_map/local_files_only raises TypeError.
        def __init__(self, *, tokenizer, audio_tokenizer, model_config):
            self.tokenizer = tokenizer
            self.audio_tokenizer = audio_tokenizer
            self.model_config = model_config

    dynamic.get_class_from_dynamic_module = Mock(return_value=StrictProcessor)
    monkeypatch.setitem(sys.modules, 'transformers', transformers)
    monkeypatch.setitem(sys.modules, 'transformers.dynamic_module_utils', dynamic)
    result = load_local_processor(str(tmp_path), 'local-codec', device, dtype)
    assert result.tokenizer is tokenizer
    assert result.audio_tokenizer is codec
    assert result.model_config is config
    local = dict(trust_remote_code=True, local_files_only=True)
    transformers.AutoConfig.from_pretrained.assert_called_once_with(str(tmp_path), **local)
    transformers.AutoTokenizer.from_pretrained.assert_called_once_with(str(tmp_path), **local)
    transformers.AutoModel.from_pretrained.assert_called_once_with(
        'local-codec', torch_dtype=dtype, device_map={'': device}, **local)
    dynamic.get_class_from_dynamic_module.assert_called_once_with(
        'processing_moss_tts.MossTTSDelayProcessor', str(tmp_path), local_files_only=True)
