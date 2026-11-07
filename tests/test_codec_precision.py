"""Real PyTorch regression for MOSS's FP32 activations / BF16 Conv1d bias."""
import pytest

torch = pytest.importorskip('torch')
from tts_lab.moss import codec_precision_context


class TinyCodec(torch.nn.Module):
    def __init__(self, dtype):
        super().__init__()
        self.out_proj = torch.nn.utils.parametrizations.weight_norm(
            torch.nn.Conv1d(4, 4, kernel_size=1)).to(dtype=dtype)

    def forward(self, values):
        # Same explicit cast immediately before out_proj as MOSS decode_code.
        return self.out_proj(values.float()).float()


def test_reproduce_reported_error_and_fix_with_real_conv():
    codec = TinyCodec(torch.bfloat16).eval()
    values = torch.randn(1, 4, 8)
    with torch.inference_mode():
        with pytest.raises(RuntimeError, match='Input type.*bias type'):
            codec(values)
        with codec_precision_context(codec):
            assert torch.is_autocast_enabled('cpu')
            output = codec(values)
        assert not torch.is_autocast_enabled('cpu')
    assert output.shape == values.shape
    assert output.dtype == torch.float32
    assert torch.isfinite(output).all()
    assert next(codec.parameters()).dtype == torch.bfloat16


def test_float32_cpu_path_stays_full_precision():
    codec = TinyCodec(torch.float32).eval()
    values = torch.randn(1, 4, 8)
    with torch.inference_mode(), codec_precision_context(codec):
        assert not torch.is_autocast_enabled('cpu')
        actual = codec(values)
    with torch.inference_mode():
        torch.testing.assert_close(actual, codec(values))


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA hardware not available')
def test_cuda_bfloat16_codec():
    codec = TinyCodec(torch.bfloat16).cuda().eval()
    values = torch.randn(1, 4, 8, device='cuda')
    with torch.inference_mode(), codec_precision_context(codec):
        assert torch.is_autocast_enabled('cuda')
        result = codec(values)
    assert torch.isfinite(result).all()
