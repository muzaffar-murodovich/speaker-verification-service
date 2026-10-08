"""Speaker embeddings from the pretrained WeSpeaker ECAPA-TDNN (VoxCeleb2 dev only)."""

from pathlib import Path

import kaldi_native_fbank as knf
import numpy as np
import onnxruntime as ort
from numpy.typing import NDArray

from svs.extract import SAMPLE_RATE

MODEL_PATH = Path("artifacts/models/voxceleb_ECAPA512_LM.onnx")
EMBEDDING_DIM = 192


def compute_fbank(samples: NDArray[np.float32]) -> NDArray[np.float32]:
    """80-bin Kaldi fbank with mean normalisation, matching WeSpeaker's ONNX example.

    WeSpeaker scales waveforms to the int16 range before fbank, so we do too.
    """
    opts = knf.FbankOptions()
    opts.frame_opts.dither = 0.0
    opts.frame_opts.window_type = "hamming"
    opts.mel_opts.num_bins = 80
    fbank = knf.OnlineFbank(opts)
    fbank.accept_waveform(SAMPLE_RATE, (samples * 32768).tolist())
    fbank.input_finished()
    if not fbank.num_frames_ready:
        raise ValueError("Audio is shorter than one 25 ms frame")
    feats = np.stack([fbank.get_frame(i) for i in range(fbank.num_frames_ready)])
    return (feats - feats.mean(axis=0)).astype(np.float32)


class Embedder:
    """Runs the ONNX model on CUDA when available, otherwise on CPU."""

    def __init__(self, model_path: str | Path = MODEL_PATH) -> None:
        available = ort.get_available_providers()
        providers = [p for p in ("CUDAExecutionProvider", "CPUExecutionProvider") if p in available]
        self.session = ort.InferenceSession(str(model_path), providers=providers)

    def embed(self, samples: NDArray[np.float32]) -> NDArray[np.float32]:
        """Return one L2-normalised embedding for mono 16 kHz speech (SVS guide §4.1)."""
        feats = compute_fbank(samples)[np.newaxis]
        embedding = self.session.run(["embs"], {"feats": feats})[0][0]
        return embedding / np.linalg.norm(embedding)
